import itertools
import json
import logging
import os
import re
import sys
from typing import Optional

import awkward as ak
import h5py
import hist
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import mplhep as hep
import numba as nb
import numpy as np
import vector
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve, roc_auc_score
vector.register_awkward()

from merged import parse_merged_w_target

hep.style.use(hep.style.ROOT)
logging.basicConfig(level=logging.INFO)

PLOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "v12vanilla_min1valid_roc_curves")
os.makedirs(PLOT_DIR, exist_ok=True)

BASE_TPR = np.linspace(0., 1., 1000)
NTOPS = 2
NJETS = 3*NTOPS + 4
NFJETS = NTOPS + 1

with open(sys.argv[1], 'r') as f:
    FILES = json.load(f)

#########################################

def build_LUT(unique_reco: str, dataset: dict[str, h5py._hl.files.File]):
    return parse_merged_w_target(dataset['test'], dataset['eval'], reco_regex=unique_reco)
def build_and_save_LUT(label: str, unique_reco: str, dataset: dict[str, h5py._hl.files.File]):
    filepath = os.path.join(PLOT_DIR, f"{label}_{unique_reco}.npy")
    if os.path.exists(filepath): return filepath
    recoLUT = build_LUT(unique_reco, dataset)
    np.save(filepath, recoLUT)
    return filepath
def read_LUT(filepath: str):
    recoLUT = None
    with open(filepath, 'rb') as f:
        recoLUT = np.load(f)
    return recoLUT

N_EVTS = {}
RECO_CLASSES = None
UNIQUE_RECOS = None
LUTS = {}
for i, (label, filepaths) in enumerate(FILES.items()):
    dataset = {filetype: h5py.File(filepath, 'a') for filetype, filepath in filepaths.items()}
    if i == 0:
        RECO_CLASSES = sorted(list(dataset["eval"]["SpecialKey.Targets"].keys()))
        UNIQUE_RECOS = {
            unique_reco: [reco for reco in RECO_CLASSES if unique_reco in reco] 
            for unique_reco in sorted(set(item[:re.search('t\d', item).start()] for item in RECO_CLASSES))
        } | {'': RECO_CLASSES}
    N_EVTS[label] = len(dataset['test']['INPUTS']['Jets']['pt'])
    LUTS[label] = {
        unique_reco: build_and_save_LUT(label, unique_reco, dataset)
        for unique_reco in UNIQUE_RECOS.keys()
    }
    for file in dataset.values(): file.close()

var_color_map = {
    'detection_probability': '#3f90da',
    'assignment_probability': '#ffa90e', 
    'marginal_probability': '#bd1f01'
}
var_legend_map = {
    'detection_probability': r'$P_{detection}$',
    'assignment_probability': r'$P_{assignment}$', 
    'marginal_probability': r'$P_{detection} \times P_{assignment}$'
}
ncorrect_linestyle_map = {
    'dashed': (0, (5, 5)),
    'singledotted': (0, (1, 5)),
    'doubledotted': (0, (1, 1, 1, 5)),
    'dashsingledotted': (0, (3, 5, 1, 5)),
    'dashdoubledotted': (0, (3, 5, 1, 1, 1, 5)),
    'singledoubledotted': (0, (1, 5, 1, 1, 1, 5)),
    'dashsingledoubledotted': (0, (3, 5, 1, 5, 1, 1, 1, 5)),
}

#########################################

def n_alpha(string: str):
    return len([c for c in string if c.isalpha()])

def find_correct(recoLUT: np.ndarray):
    return recoLUT[..., 1].astype("bool")
def find_n_correct(recoLUT: np.ndarray):
    return np.sum(find_correct(recoLUT), axis=1)

def find_valid(recoLUT: np.ndarray):
    return recoLUT[..., 0].astype("bool")
def find_n_valid(recoLUT: np.ndarray):
    return np.sum(find_valid(recoLUT), axis=1)

def merge_evt_pred_masks(full_arr: np.ndarray, evt_mask: Optional[np.ndarray]=None, pred_mask: Optional[np.ndarray]=None):
    return np.logical_and(np.tile(evt_mask, (full_arr.shape[1], 1)).T, pred_mask)

def get_preds(recoLUT: np.ndarray, score: str, evt_mask: np.ndarray=None, pred_mask: np.ndarray=None):
    if score == "detection_probability": full_preds = recoLUT[..., 6]
    elif score == "assignment_probability": full_preds = recoLUT[..., 7]
    elif score == "marginal_probability": full_preds = recoLUT[..., 6] * recoLUT[..., 7]
    else: raise Exception(f"Provided score not in the known list: {score}")
    if evt_mask is None: evt_mask = np.ones_like(full_preds[0], dtype=bool)
    if pred_mask is None: pred_mask = np.ones_like(full_preds, dtype=bool)
    full_mask = np.logical_and(merge_evt_pred_masks(full_preds, evt_mask, pred_mask), recoLUT[..., 3].astype("bool"))
    return full_preds[full_mask]

def get_truths(recoLUT: np.ndarray, evt_mask: np.ndarray=None, pred_mask: np.ndarray=None):
    full_truths = recoLUT[..., 0]
    if evt_mask is None: evt_mask = np.ones_like(full_truths[0], dtype=bool)
    if pred_mask is None: pred_mask = np.ones_like(full_truths, dtype=bool)
    full_mask = merge_evt_pred_masks(full_truths, evt_mask, pred_mask)
    return full_truths[full_mask]

#########################################

def interp_roc_curve(truths, preds, **kwargs):
    fpr, tpr, thresholds = roc_curve(truths, preds, **kwargs)
    interp_fpr = np.interp(BASE_TPR, tpr, fpr)
    interp_thresholds = np.interp(BASE_TPR, tpr, thresholds)
    return interp_fpr, BASE_TPR, interp_thresholds

def get_auc(fpr, tpr):
    return 1 - trapezoid(fpr, x=tpr)

def plot_roc(fprs: list[np.ndarray], tprs: list[np.ndarray], labels: list[str], colors: Optional[list[str]]=None, linestyles: Optional[list[str]]=None, vlines: Optional[list[float]]=None, hlines: Optional[list[float]]=None, title: Optional[str]=None, xlabel: Optional[str]=None, ylabel: Optional[str]=None, save: Optional[str]=None, show: bool=False):
    if xlabel is None: xlabel = 'Bkg Eff.'
    if ylabel is None: ylabel = 'Sig Eff.'

    plt.figure(figsize=(21, 10))
    if colors is None: 
        colors = [None] * len(fprs)
    if linestyles is None:
        linestyles = ['solid'] * len(fprs)
    for fpr, tpr, label, color, linestyle in zip(fprs, tprs, labels, colors, linestyles):
        plt.plot(fpr, tpr, label=label, color=color, linestyle=linestyle)
    if vlines is not None: 
        for vli, vla in vlines:
            plt.vlines(vli, label=vla, xmin=np.min(tprs, axis=None), xmax=np.max(tprs, axis=None))
    if hlines is not None: 
        for hli, hla in hlines:
            plt.hlines(hli, label=hla, xmin=np.min(fprs, axis=None), xmax=np.max(fprs, axis=None))
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.yscale('log')
    if title is not None: plt.title(title)
    plt.tight_layout()
    if save is not None: plt.savefig(os.path.join(PLOT_DIR, save))
    if show: plt.show()
    else: plt.close()

def make_rocs(unique_reco: str, rocs: list[dict], **kwargs):
    fprs, tprs, labels, colors, linestyles = [], [], [], [], []
    for roc in rocs:

        for var in roc['vars']:
            color = roc['color']
            linestyle = roc['linestyle']

            preds, truths = [], []
            for label in ['sig', 'bkg']:
                if label not in ['sig', 'bkg']: continue

                for subdata in roc[label]:
                    recoLUT = read_LUT(LUTS[subdata['dataset']][unique_reco])

                    evt_mask = np.zeros(N_EVTS[subdata['dataset']], dtype=bool)
                    for mask in subdata['masks']:
                        for n_valid, n_corrects in mask.items():
                            if type(n_corrects) is int: n_corrects = [n_corrects]
                            assert type(n_corrects) is list, f'You must pass either an int or a list to the n_corrects value in the mask dictionaries. You provided {type(n_corrects)}'
                            n_valid_mask = (find_n_valid(recoLUT) == n_valid)
                            n_correct_mask = np.isin(find_n_correct(recoLUT), n_corrects)
                            evt_mask = np.logical_or(evt_mask, np.logical_and(n_valid_mask, n_correct_mask))
                    pred_mask = np.isin(find_correct(recoLUT).astype("bool"), subdata["corrs"])
                    subpreds = get_preds(recoLUT, var, evt_mask=evt_mask, pred_mask=pred_mask)
                    subtruths = np.ones_like(subpreds, dtype=int) if label == 'sig' else np.zeros_like(subpreds, dtype=int)
                    preds.append(subpreds); truths.append(subtruths)

            fpr, tpr, threshold = interp_roc_curve(np.concatenate(truths, axis=0), np.concatenate(preds, axis=0), **kwargs)
            fprs.append(fpr); tprs.append(tpr); labels.append(' - '.join([roc['label'], var_legend_map[var], f'AUC = {get_auc(fpr, tpr):.3f}'])); colors.append(color), linestyles.append(linestyle)
    return fprs, tprs, labels, colors, linestyles

#########################################
#########################################

tt_012valid_vs_qcd_rocs = [
    {
        'vars': ['detection_probability'], 
        'color': '#3f90da', 'linestyle': ncorrect_linestyle_map['dashsingledoubledotted'], 'label': 'All tops, 2 valid tt',
        'sig': [{'masks': [{2: [0, 1, 2]}], 'corrs': [True, False], 'dataset': "tt"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#ffa90e', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 2 valid tt',
        'sig': [{'masks': [{2: [1, 2]}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}, {'masks': [{2: [0, 1]}], 'corrs': [False], 'dataset': "tt"}],
    },
    {
        'vars': ['detection_probability'], 
        'color': '#bd1f01', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 1 valid tt',
        'sig': [{'masks': [{1: [0, 1]}], 'corrs': [True, False], 'dataset': "tt"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#94a4a2', 'linestyle': ncorrect_linestyle_map['singledotted'], 'label': 'Correct tops, 1 valid tt',
        'sig': [{'masks': [{1: 1}], 'corrs': [True], 'dataset': "t"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}, {'masks': [{1: 0}], 'corrs': [False], 'dataset': "tt"}],
    },
    {
        'vars': ['detection_probability'], 
        'color': '#832db6', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 0 valid tt',
        'sig': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "tt"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, tt_012valid_vs_qcd_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of tt vs. QCD - {unique_reco} Category", save=os.path.join(PLOT_DIR, f'tt_012valid_vs_qcd_rocs{unique_reco}.png'))

#########################################

tt_12valid_corr_vs_incorr_rocs = [
    {
        'vars': ['marginal_probability'], 
        'color': '#92dadd', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 2 valid tt',
        'sig': [{'masks': [{2: [1, 2]}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{2: [0, 1]}], 'corrs': [False], 'dataset': "tt"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#e76300', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 1 valid tt',
        'sig': [{'masks': [{1: 1}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: 0}], 'corrs': [False], 'dataset': "tt"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, tt_12valid_corr_vs_incorr_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct tt vs. Incorrect tt - {unique_reco} Category", save=os.path.join(PLOT_DIR, f'tt_12valid_corr_vs_incorr_rocs{unique_reco}.png'))

#########################################

tt_12valid_corr_vs_incorr_rocs = [
    {
        'vars': ['marginal_probability'], 
        'color': '#92dadd', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 2 valid tt',
        'sig': [{'masks': [{2: [1, 2]}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{2: [0, 1]}], 'corrs': [False], 'dataset': "tt"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#e76300', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 1 valid tt',
        'sig': [{'masks': [{1: 1}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: 0}], 'corrs': [False], 'dataset': "tt"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, tt_12valid_corr_vs_incorr_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct tt vs. Incorrect tt - {unique_reco} Category", save=os.path.join(PLOT_DIR, f'tt_12valid_corr_vs_incorr_rocs{unique_reco}.png'))

#########################################

t_01valid_vs_qcd_rocs = [
    {
        'vars': ['detection_probability'], 
        'color': '#bd1f01', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 1 valid t',
        'sig': [{'masks': [{1: [0, 1]}], 'corrs': [True, False], 'dataset': "t"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#94a4a2', 'linestyle': ncorrect_linestyle_map['singledotted'], 'label': 'Correct tops, 1 valid t',
        'sig': [{'masks': [{1: 1}], 'corrs': [True], 'dataset': "t"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}, {'masks': [{1: 0}], 'corrs': [False], 'dataset': "t"}],
    },
    {
        'vars': ['detection_probability'], 
        'color': '#832db6', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 0 valid t',
        'sig': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "t"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "qcd"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, t_01valid_vs_qcd_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of t vs. QCD - {unique_reco} Category", save=os.path.join(PLOT_DIR, f't_01valid_vs_qcd_rocs{unique_reco}.png'))

#########################################

t_1valid_corr_vs_incorr_rocs = [
    {
        'vars': ['marginal_probability'], 
        'color': '#e76300', 'linestyle': ncorrect_linestyle_map['doubledotted'], 'label': 'Correct tops, 1 valid t',
        'sig': [{'masks': [{1: 1}], 'corrs': [True], 'dataset': "t"}],
        'bkg': [{'masks': [{1: 0}], 'corrs': [False], 'dataset': "t"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, t_1valid_corr_vs_incorr_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct t vs. Incorrect t - {unique_reco} Category", save=os.path.join(PLOT_DIR, f't_1valid_corr_vs_incorr_rocs{unique_reco}.png'))

#########################################

tt_012valid_vs_t_01valid_rocs = [
    {
        'vars': ['detection_probability'], 
        'color': '#717581', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 2 valid tt, 1 valid t',
        'sig': [{'masks': [{2: [0, 1, 2]}], 'corrs': [True, False], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: [0, 1]}], 'corrs': [True, False], 'dataset': "t"}],
    },
    {
        'vars': ['detection_probability'], 
        'color': '#b9ac70', 'linestyle': ncorrect_linestyle_map['dashsingledotted'], 'label': 'All tops, 1 valid tt, 1 valid t',
        'sig': [{'masks': [{1: [0, 1]}], 'corrs': [True, False], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: [0, 1]}], 'corrs': [True, False], 'dataset': "t"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#e76300', 'linestyle': ncorrect_linestyle_map['singledotted'], 'label': 'Correct tops, 2 valid tt, 1 valid t',
        'sig': [{'masks': [{2: [0, 1, 2]}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: [0, 1]}], 'corrs': [True], 'dataset': "t"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#964a8b', 'linestyle': ncorrect_linestyle_map['singledotted'], 'label': 'Correct tops, 1 valid tt, 1 valid t',
        'sig': [{'masks': [{1: [0, 1]}], 'corrs': [True], 'dataset': "tt"}],
        'bkg': [{'masks': [{1: [0, 1]}], 'corrs': [True], 'dataset': "t"}],
    },
    {
        'vars': ['marginal_probability'], 
        'color': '#e42536', 'linestyle': ncorrect_linestyle_map['dashed'], 'label': 'Incorrect tops, 0 valid tt, 0 valid t',
        'sig': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "tt"}],
        'bkg': [{'masks': [{0: 0}], 'corrs': [False], 'dataset': "t"}],
    },
]
for unique_reco in UNIQUE_RECOS.keys():
    fprs, tprs, labels, colors, linestyles = make_rocs(unique_reco, tt_012valid_vs_t_01valid_rocs)
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of tt vs. t - {unique_reco} Category", save=os.path.join(PLOT_DIR, f'tt_012valid_vs_t_01valid_rocs{unique_reco}.png'))

#########################################
    