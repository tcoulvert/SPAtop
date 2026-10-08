import itertools
import logging
import os
import re
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

# from utils import get_jets, get_numerical, get_symmetries, match_jet, reset_collision_dp
# from merged import sel_pred_t_by_prob
from utils import get_symmetries

hep.style.use(hep.style.ROOT)
logging.basicConfig(level=logging.INFO)

PLOT_DIR = os.path.join(os.getcwd(), "multi_roc_curves")
os.makedirs(PLOT_DIR, exist_ok=True)

BASE_TPR = np.linspace(0., 1., 1000)
NTOPS = 2
NJETS = 3*NTOPS + 4
NFJETS = NTOPS + 1

FILES = {
    "tt": {"eval": "/storage/af/user/tsievert/topNet/h5s/tt_hadronic_noBQ.h5", "test": "/storage/af/user/tsievert/topNet/h5s/fjTag_testing.h5"},
    "qcd": {"eval": "/storage/af/user/tsievert/topNet/h5s/qcd_4j_Alltesting_noBQeval.h5", "test": "/storage/af/user/tsievert/topNet/h5s/qcd_4j_wTargets_Alltesting.h5"},
    "t": {"eval": "/storage/af/user/tsievert/topNet/h5s/tt_semileptonic_Alltesting_noBQeval.h5", "test": "/storage/af/user/tsievert/topNet/h5s/ttbar_semileptonic/ttbar_semileptonic.h5"}
    # "4t": "",
}

#########################################

LOADED_FILES = {label: {filetype: h5py.File(filepath) for filetype, filepath in filepaths.items()} for label, filepaths in FILES.items()}
N_EVTS = {label: len(filepaths['test']['INPUTS']['Jets']['pt']) for label, filepaths in LOADED_FILES.items()}
RECO_CLASSES = sorted(list(LOADED_FILES[next(iter(LOADED_FILES))]["eval"]["SpecialKey.Targets"].keys()))
UNIQUE_RECOS = {unique_reco: [reco for reco in RECO_CLASSES if unique_reco in reco] for unique_reco in sorted(set(item[:re.search('t\d', item).start()] for item in RECO_CLASSES))}
SCORES = [item for item in LOADED_FILES[next(iter(LOADED_FILES))]["eval"]["SpecialKey.Targets"][next(iter(RECO_CLASSES))].keys() if "probability" in item]
ASSIGNMENTS = {unique_reco: [item for item in LOADED_FILES[next(iter(LOADED_FILES))]["eval"]["SpecialKey.Targets"][next(iter(UNIQUE_RECOS[unique_reco]))].keys() if item not in SCORES] for unique_reco in UNIQUE_RECOS.keys()}
SYMMETRIES = dict(zip(UNIQUE_RECOS.keys(), get_symmetries(UNIQUE_RECOS.keys(), ASSIGNMENTS, returnidx=False)))

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

def find_correct(unique_reco: str, dataset: dict[str, str]):
    return np.array([
        np.any([
            np.any([
                np.all([
                    (
                        dataset['eval']['SpecialKey.Targets'][eval_reco][assn[0]][:] == dataset['test']['TARGETS'][test_reco][assn[1]][:]
                        if n_alpha(assn[0]) == 1 else
                        dataset['eval']['SpecialKey.Targets'][eval_reco][assn[0]][:] == (dataset['test']['TARGETS'][test_reco][assn[1]][:] + NJETS)
                    ) for assn in symmetry
                ], axis=0)
                for symmetry in SYMMETRIES[unique_reco]
            ], axis=0)
            for test_reco in UNIQUE_RECOS[unique_reco]
        ], axis=0)
        for eval_reco in UNIQUE_RECOS[unique_reco]
    ]).astype("bool")
def find_n_correct(unique_reco: str, dataset: dict[str, str]):
    return np.sum(find_correct(unique_reco, dataset), axis=0)

def find_valid(unique_reco: str, dataset: dict[str, str]):
    return np.array([
        dataset['test']['TARGETS'][test_reco]['MASK'][:]
        for test_reco in UNIQUE_RECOS[unique_reco]
    ]).astype("bool")
def find_n_valid(unique_reco: str, dataset: dict[str, str]):
    return np.sum(find_valid(unique_reco, dataset), axis=0)

def merge_evt_pred_masks(unique_reco: str, evt_mask: Optional[np.ndarray]=None, pred_mask: Optional[np.ndarray]=None):
    return np.logical_and(np.array([evt_mask for _ in UNIQUE_RECOS[unique_reco]]), pred_mask)

def get_preds(unique_reco: str, dataset: dict[str, str], score: str, evt_mask: np.ndarray=None, pred_mask: np.ndarray=None):
    full_preds = np.array([dataset['eval']['SpecialKey.Targets'][reco][score] for reco in UNIQUE_RECOS[unique_reco]])
    if evt_mask is None: evt_mask = np.ones_like(full_preds[0], dtype=bool)
    if pred_mask is None: pred_mask = np.ones_like(full_preds, dtype=bool)
    full_mask = merge_evt_pred_masks(unique_reco, evt_mask, pred_mask)
    return full_preds[full_mask]

def get_truths(unique_reco: str, dataset: dict[str, str], evt_mask: np.ndarray=None, pred_mask: np.ndarray=None):
    full_truths = np.array([dataset['eval']['TARGETS'][reco]['MASK'] for reco in UNIQUE_RECOS[unique_reco]])
    if evt_mask is None: evt_mask = np.ones_like(full_truths[0], dtype=bool)
    if pred_mask is None: pred_mask = np.ones_like(full_truths, dtype=bool)
    full_mask = merge_evt_pred_masks(unique_reco, evt_mask, pred_mask)
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

    plt.figure(figsize=(10, 8))
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
    plt.tight_layout()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.yscale('log')
    if title is not None: plt.title(title)
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
                    dataset = LOADED_FILES[subdata['dataset']]

                    evt_mask = np.zeros(N_EVTS[subdata['dataset']], dtype=bool)
                    for mask in subdata['masks']:
                        for n_valid, n_corrects in mask.items():
                            if type(n_corrects) is int: n_corrects = [n_corrects]
                            assert type(n_corrects) is list, f'You must pass either an int or a list to the n_corrects value in the mask dictionaries. You provided {type(n_corrects)}'
                            n_valid_mask = (find_n_valid(unique_reco, dataset) == n_valid)
                            n_correct_mask = np.isin(find_n_correct(unique_reco, dataset), n_corrects)
                            evt_mask = np.logical_or(evt_mask, np.logical_and(n_valid_mask, n_correct_mask))
                    pred_mask = np.isin(find_correct(unique_reco, dataset).astype("bool"), subdata["corrs"])
                    subpreds = get_preds(unique_reco, dataset, var, evt_mask=evt_mask, pred_mask=pred_mask)
                    subtruths = np.ones_like(subpreds) if label == 'sig' else np.zeros_like(subpreds)
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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of tt vs. QCD - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f'tt_012valid_vs_qcd_roc_{unique_reco}.png'))

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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct tt vs. Incorrect tt - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f'tt_12valid_corr_vs_incorr_rocs{unique_reco}.png'))

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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct tt vs. Incorrect tt - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f'tt_12valid_corr_vs_incorr_rocs{unique_reco}.png'))

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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of t vs. QCD - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f't_01valid_vs_qcd_rocs{unique_reco}.png'))

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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of Correct t vs. Incorrect t - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f't_1valid_corr_vs_incorr_rocs{unique_reco}.png'))

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
    plot_roc(fprs, tprs, labels, colors=colors, linestyles=linestyles, title=f"Probabilities of tt vs. t - {unique_reco} Category", show=True, save=os.path.join(PLOT_DIR, f'tt_012valid_vs_t_01valid_rocs{unique_reco}.png'))

#########################################

for files in LOADED_FILES.values(): 
    for file in files.values(): file.close()
    