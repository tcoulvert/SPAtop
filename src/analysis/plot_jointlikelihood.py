import copy
import os

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

import awkward as ak
import vector as vec
vec.register_awkward()

from src.analysis.boosted import parse_boosted_w_target
from src.analysis.resolved import parse_resolved_w_target
from src.analysis.semi_resolved import parse_semi_resolved_w_target
from src.analysis.utils import calc_eff, calc_pur


def detass_probability(pred_h5, reco_class: str):
    return (
        pred_h5["SpecialKey.Targets"][reco_class]["detection_probability"][:]
        * pred_h5["SpecialKey.Targets"][reco_class]["assignment_probability"][:]
    )
def notexist_probability(pred_h5, reco_class: str):
    return 1 - pred_h5["SpecialKey.Targets"][reco_class]["detection_probability"][:]

def log_likelihood(detass_prob, notexist_prob):
    return np.log(detass_prob) - np.log(notexist_prob)


def reco_toppt(pred_h5, jet_assn_keys: dict[str, np.ndarray]):
    Jets = ak.from_regular(ak.zip({
        "pt": pred_h5['INPUTS']['Jets']['pt'][:],
        "eta": pred_h5['INPUTS']['Jets']['eta'][:],
        "phi": pred_h5['INPUTS']['Jets']['phi'][:],
        "mass": pred_h5['INPUTS']['Jets']['mass'][:],
    }, with_name="Momentum4D"))
    BoostedJets = ak.from_regular(ak.zip({
        "pt": pred_h5['INPUTS']['BoostedJets']['fj_pt'][:],
        "eta": pred_h5['INPUTS']['BoostedJets']['fj_eta'][:],
        "phi": pred_h5['INPUTS']['BoostedJets']['fj_phi'][:],
        "mass": pred_h5['INPUTS']['BoostedJets']['fj_mass'][:],
    }, with_name="Momentum4D"))
    n_jets = pred_h5['INPUTS']['Jets']['pt'][:].shape[1]
    n_fatjets = pred_h5['INPUTS']['BoostedJets']['fj_pt'][:].shape[1]

    jets = []
    for key, value in jet_assn_keys.items():
        if len([k for k in key if k.isalpha()]) == 1:
            jets.append(ak.firsts(Jets[ak.local_index(Jets) == value]))
        else:
            jets.append(ak.firsts(BoostedJets[ak.local_index(BoostedJets) == (value - n_jets)]))

    reco_top = sum(jets)
    return reco_top.pt


def split_pur_eff_toppts(target_h5, pred_h5, reco_class: str, reco_classes_toppts: dict):
    jet_assignment_keys = [key for key in pred_h5["SpecialKey.Targets"][reco_class].keys() if 'probability' not in key]

    per_event_predictions = np.array([pred_h5["SpecialKey.Targets"][reco_class][key][:] for key in jet_assignment_keys]).T
    per_event_loglikelihood = log_likelihood(
        np.array(detass_probability(pred_h5, reco_class)), np.array(notexist_probability(pred_h5, reco_class))
    )
    pred_mask = (per_event_loglikelihood > 0)
    per_event_targets = np.array([target_h5["TARGETS"][reco_class][key][:] for key in jet_assignment_keys]).T
    target_mask = target_h5["TARGETS"][reco_class]['MASK'][:]

    correct_mask = np.logical_and(target_mask, np.all(per_event_predictions == per_event_targets, axis=1))

    recotoppt = reco_toppt(pred_h5, {key: per_event_predictions[:, i] for i, key in enumerate(jet_assignment_keys)})
    gentoppt = target_h5["TARGETS"][reco_class]['pt'][:]

    reco_classes_toppts['correct_and_found_recopt'].append(recotoppt[np.logical_and(correct_mask, pred_mask)])
    reco_classes_toppts['all_found_recopt'].append(recotoppt[pred_mask])
    reco_classes_toppts['correct_and_found_genpt'].append(gentoppt[np.logical_and(correct_mask, pred_mask)])
    reco_classes_toppts['all_correct_genpt'].append(gentoppt[target_mask])

def merged_pur_eff_toppts(target_h5, pred_h5, reco_classes: list[str], reco_classes_toppts: dict, chosen_recos: ak.Array):
    jet_assignment_keys = [key for key in pred_h5["SpecialKey.Targets"][reco_class].keys() if 'probability' not in key]

    per_event_predictions = np.array([pred_h5["SpecialKey.Targets"][reco_class][key][:] for key in jet_assignment_keys]).T
    per_event_loglikelihood = log_likelihood(
        np.array(detass_probability(pred_h5, reco_class)), np.array(notexist_probability(pred_h5, reco_class))
    )
    pred_mask = (per_event_loglikelihood > 0)
    per_event_targets = np.array([target_h5["TARGETS"][reco_class][key][:] for key in jet_assignment_keys]).T
    target_mask = target_h5["TARGETS"][reco_class]['MASK'][:]

    correct_mask = np.logical_and(target_mask, np.all(per_event_predictions == per_event_targets, axis=1))

    recotoppt = reco_toppt(pred_h5, {key: per_event_predictions[:, i] for i, key in enumerate(jet_assignment_keys)})
    gentoppt = target_h5["TARGETS"][reco_class]['pt'][:]

    reco_classes_toppts['correct_and_found_recopt'].append(recotoppt[np.logical_and(correct_mask, pred_mask)])
    reco_classes_toppts['all_found_recopt'].append(recotoppt[pred_mask])
    reco_classes_toppts['correct_and_found_genpt'].append(gentoppt[np.logical_and(correct_mask, pred_mask)])
    reco_classes_toppts['all_correct_genpt'].append(gentoppt[target_mask])
    

def calc_pur_eff(target_path, pred_path, bins_dict, chi2_cuts=[45, 20]):
    # open files
    target_h5 = h5.File(target_path)
    pred_h5 = h5.File(pred_path)

    max_tops = max([key[-1] for key in pred_h5["SpecialKey.Targets"].keys()])
    reco_classes = list(set([key[:-1] for key in pred_h5["SpecialKey.Targets"].keys()]))

    pur_eff_toppts = {
        reco_class: copy.deepcopy({
            'correct_and_found_recopt': [], 'all_found_recopt': [], 
            'correct_and_found_genpt': [], 'all_correct_genpt': []
        })
        for reco_class in reco_classes+['Merged']
    }
    for top_idx in range(1, max_tops+1):
        pred_options = [reco_class+str(top_idx) for reco_class in reco_classes]

        for pred_option, reco_class in zip(pred_options, reco_classes):
            split_pur_eff_toppts(target_h5, pred_h5, pred_option, pur_eff_toppts[reco_class])

    # Merged
    all_pred_options = [reco_class+str(top_idx) for reco_class in reco_classes for top_idx in range(1, max_tops+1)]
    exists_and_correct = np.array([detass_probability(pred_h5, pred_option) for pred_option in all_pred_options]).T
    doesnt_exist = np.array([notexist_probability(pred_h5, pred_option) for pred_option in all_pred_options]).T
    loglikelihood = log_likelihood(exists_and_correct, doesnt_exist)
    chosen_options = ak.from_regular(np.tile(all_pred_options, (loglikelihood.shape(0), 1))[loglikelihood.argsort(axis=1, descending=True)])[loglikelihood > 0]

    for pred_option in all_pred_options:
        merged_pur_eff_toppts(target_h5, pred_h5, pred_option, pur_eff_toppts['Merged'], chosen_options)

    ## Make pur/eff plots ##