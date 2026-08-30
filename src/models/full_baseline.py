import copy
import os

import h5py
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import awkward as ak
import vector as vec
vec.register_awkward()

N_TOPS = 2
TOP_MASS = 172.52  # GeV
TOP_SIGMA = 20.
W_MASS = 80.37  # GeV
W_SIGMA = 14.

AK8_SIZE = 0.8
AK5_SIZE = 0.5

FILL_VALUE = 1e5

# PLOT_CHI2_HISTS = False
# PLOT_ROCS = False
SAVE_H5 = True

FILEPATH = os.path.abspath(__file__)
DIRPATH = '/'.join(FILEPATH.split('/')[:-1])
PLOT_DIRPATH = os.path.join(DIRPATH, f"v11p2/FullAnalysisEmu")
if not os.path.exists(PLOT_DIRPATH): os.makedirs(PLOT_DIRPATH)

file_path = "/storage/af/user/tsievert/topNet/fjTag_testing.h5"
################################################
# 1) Load arrays
################################################
with h5py.File(file_path, "r") as f:
    mask = f['INPUTS/Jets/MASK'][:]
    pt   = f['INPUTS/Jets/pt'][:]
    eta  = f['INPUTS/Jets/eta'][:]
    phi  = f['INPUTS/Jets/phi'][:]
    mass = f['INPUTS/Jets/mass'][:]
    btag = f['INPUTS/Jets/btag'][:]
    
    fj_mask = f['INPUTS/BoostedJets/MASK'][:]
    fj_pt   = f['INPUTS/BoostedJets/fj_pt'][:]
    fj_eta  = f['INPUTS/BoostedJets/fj_eta'][:]
    fj_phi  = f['INPUTS/BoostedJets/fj_phi'][:]
    fj_mass = f['INPUTS/BoostedJets/fj_mass'][:]
    fj_Ttag = f['INPUTS/BoostedJets/fj_Ttag'][:]
    fj_Wtag = f['INPUTS/BoostedJets/fj_Wtag'][:]

    # FR
    tgt_FR_dict = {
        'FRt1/b': f['TARGETS/FRt1/b'][:],
        'FRt1/q1': f['TARGETS/FRt1/q1'][:],
        'FRt1/q2': f['TARGETS/FRt1/q2'][:],
        'FRt1/mask': f["TARGETS/FRt1/MASK"][:],

        'FRt2/b': f['TARGETS/FRt2/b'][:],
        'FRt2/q1': f['TARGETS/FRt2/q1'][:],
        'FRt2/q2': f['TARGETS/FRt2/q2'][:],
        'FRt2/mask': f["TARGETS/FRt2/MASK"][:],
    }

    # SRqq
    tgt_SRqq_dict = {
        'SRqqt1/b': f['TARGETS/SRqqt1/b'][:],
        'SRqqt1/qq': f['TARGETS/SRqqt1/qq'][:],
        'SRqqt1/mask': f["TARGETS/SRqqt1/MASK"][:],

        'SRqqt2/b': f['TARGETS/SRqqt2/b'][:],
        'SRqqt2/qq': f['TARGETS/SRqqt2/qq'][:],
        'SRqqt2/mask': f["TARGETS/SRqqt2/MASK"][:],
    }

    # FB
    tgt_FB_dict = {
        'FBt1/bqq': f['TARGETS/FBt1/bqq'][:],
        'FBt1/mask': f["TARGETS/FBt1/MASK"][:],

        'FBt2/bqq': f['TARGETS/FBt2/bqq'][:],
        'FBt2/mask': f["TARGETS/FBt2/MASK"][:],
    }

################################################
# 2) Build jagged [events][jets] array
_jets_ = ak.from_regular(ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass,
    "btag": btag
}, with_name="Momentum4D"))
_jets_["index"] = ak.local_index(_jets_, axis=1)
evt_order = np.arange(ak.num(_jets_, axis=0))

_fjets_ = ak.from_regular(ak.zip({
    "pt": fj_pt,
    "eta": fj_eta,
    "phi": fj_phi,
    "mass": fj_mass,
    "Ttag": fj_Ttag,
    "Wtag": fj_Wtag
}, with_name="Momentum4D"))
_fjets_["index"] = ak.local_index(_fjets_, axis=1)

################################################
## Selection ##
################################################
dataset = {}
n_selected_tops = ak.zeros_like(evt_order)
################################################
# Fill Boosted events
fjets = ak.mask(_fjets_, fj_mask)
bqq_fjets = ak.drop_none(ak.mask(fjets, fjets['Ttag']), axis=1)
selected_bqq = ak.pad_none(bqq_fjets[ak.argsort(bqq_fjets.pt, axis=1, ascending=False)], N_TOPS, axis=1, clip=True)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_bqq, axis=1), axis=1)
print(f"n_selected after Boosted = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/FBt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_bqq, axis=1), axis=1) > i)
    dataset[f'TARGETS/FBt{i+1}/bqq'] = ak.fill_none(selected_bqq["index"][:, i], -1)
    dataset[f'TARGETS/FBt{i+1}/assignment_probability'] = dataset[f'TARGETS/FBt{i+1}/detection_probability']

postFB_fj_mask = copy.deepcopy(fj_mask); postFB_j_mask = copy.deepcopy(mask)
for i in range(N_TOPS):
    postFB_fj_mask = np.logical_and(
        postFB_fj_mask, 
        np.logical_and(
            _fjets_["index"] != dataset[f'TARGETS/FBt{i+1}/bqq'],
            _fjets_.deltaR(
                ak.fill_none(
                    ak.firsts(_fjets_[_fjets_["index"] == dataset[f'TARGETS/FBt{i+1}/bqq']], axis=1),
                    ak.zip({'pt': -999, 'eta': -999, 'phi': -999, 'mass': -999}, with_name="Momentum4D")
                )
            ) > AK8_SIZE
        )
    )
    postFB_j_mask = np.logical_and(
        postFB_j_mask, 
        _jets_.deltaR(
            ak.fill_none(
                ak.firsts(_fjets_[_fjets_["index"] == dataset[f'TARGETS/FBt{i+1}/bqq']], axis=1),
                ak.zip({'pt': -999, 'eta': -999, 'phi': -999, 'mass': -999}, with_name="Momentum4D")
            )
        ) > AK8_SIZE
    )

################################################
# Fill Semi-Resolved (qq) events
fjets = ak.mask(_fjets_, postFB_fj_mask)
jets = ak.mask(_jets_, postFB_j_mask)
qq_fjets = ak.drop_none(ak.mask(fjets, fjets['Wtag']), axis=1)
b_jets = ak.drop_none(ak.mask(jets, jets['btag']), axis=1)

t = ak.cartesian({'w': qq_fjets, 'b': b_jets}, axis=1)
t = ak.with_field(t, (t.w + t.b).mass, "mass")
t = ak.with_field(t, t.w.deltaR(t.b), "deltaR")
t = ak.with_field(t, ((t.mass - TOP_MASS) / TOP_SIGMA)**2, "chi2")

t_mask = (t["deltaR"] > AK8_SIZE)

selected_qq, selected_b = [], []
for i in range(N_TOPS):
    chi2_SRqq = ak.where(t_mask, t["chi2"], FILL_VALUE)
    best_idx = ak.argmin(chi2_SRqq, axis=1)
    best_chi2 = ak.firsts(chi2_SRqq[ak.local_index(chi2_SRqq) == best_idx], axis=1)
    best_t = ak.firsts(t[ak.local_index(t) == best_idx], axis=1)

    good_chi2 = ((N_TOPS - n_selected_tops) > i) & (best_chi2 != FILL_VALUE)
    selected_qq.append(ak.mask(best_t.w, good_chi2)); selected_b.append(ak.mask(best_t.b, good_chi2))

    t_mask = t_mask & (
        ((t.w["index"] != best_t.w["index"]) & (t.b["index"] != best_t.b["index"]))
        | ~good_chi2
    )
    t_mask = ak.fill_none(t_mask, [], axis=0)

selected_qq = ak.concatenate([selection[:, np.newaxis] for selection in selected_qq], axis=1)
selected_b = ak.concatenate([selection[:, np.newaxis] for selection in selected_b], axis=1)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_qq, axis=1), axis=1)
print(f"n_selected after Semi-Resolved = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/SRqqt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_qq, axis=1), axis=1) > i)
    dataset[f'TARGETS/SRqqt{i+1}/b'] = ak.fill_none(selected_b["index"][:, i], -1)
    dataset[f'TARGETS/SRqqt{i+1}/qq'] = ak.fill_none(selected_qq["index"][:, i], -1)
    dataset[f'TARGETS/SRqqt{i+1}/assignment_probability'] = dataset[f'TARGETS/SRqqt{i+1}/detection_probability']

postSR_fj_mask = copy.deepcopy(postFB_fj_mask); postSR_j_mask = copy.deepcopy(postFB_j_mask)
for i in range(N_TOPS):
    postSR_fj_mask = np.logical_and(
        postSR_fj_mask, 
        np.logical_and(
            _fjets_["index"] != dataset[f'TARGETS/SRqqt{i+1}/qq'],
            _fjets_.deltaR(
                ak.fill_none(
                    ak.firsts(_fjets_[_fjets_["index"] == dataset[f'TARGETS/SRqqt{i+1}/qq']], axis=1),
                    ak.zip({'pt': -999, 'eta': -999, 'phi': -999, 'mass': -999}, with_name="Momentum4D")
                )
            ) > AK8_SIZE
        )
    )
    postSR_j_mask = np.logical_and(
        postSR_j_mask, 
        np.logical_and(
            _jets_["index"] != dataset[f'TARGETS/SRqqt{i+1}/b'],
            np.logical_and(
                _jets_.deltaR(
                    ak.fill_none(
                        ak.firsts(_fjets_[_fjets_["index"] == dataset[f'TARGETS/SRqqt{i+1}/qq']], axis=1),
                        ak.zip({'pt': -999, 'eta': -999, 'phi': -999, 'mass': -999}, with_name="Momentum4D")
                    )
                ) > AK8_SIZE,
                _jets_.deltaR(
                    ak.fill_none(
                        ak.firsts(_jets_[_jets_["index"] == dataset[f'TARGETS/SRqqt{i+1}/b']], axis=1),
                        ak.zip({'pt': -999, 'eta': -999, 'phi': -999, 'mass': -999}, with_name="Momentum4D")
                    )
                ) > AK5_SIZE
            )
        )
    )

################################################
# Fill Fully-Resolved events
jets = ak.mask(_jets_, postSR_j_mask)
bjets = ak.drop_none(ak.mask(jets, jets['btag']), axis=1)
ljets = ak.drop_none(ak.mask(jets, ~jets['btag']), axis=1)

w = ak.combinations(ljets, 2, axis=1, fields=["j1", "j2"])
w = ak.with_field(w, (w.j1 + w.j2).mass, "mass")

t = ak.cartesian({'w': w, 'b': bjets}, axis=1)
t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).mass, "mass")
t = ak.with_field(t, t.w.j1.deltaR(t.w.j2) + t.w.j1.deltaR(t.b) + t.w.j2.deltaR(t.b), "sumdeltaR")
t = ak.with_field(t, ((t.w.mass - W_MASS) / W_SIGMA )**2 + ((t.mass - TOP_MASS) / TOP_SIGMA)**2, "chi2")

t_mask = (t['sumdeltaR'] > 3*AK5_SIZE)

selected_b, selected_q1, selected_q2 = [], [], []
for i in range(N_TOPS):
    chi2_FR = ak.where(t_mask, t["chi2"], FILL_VALUE)
    best_idx = ak.argmin(chi2_FR, axis=1)
    best_chi2 = ak.firsts(chi2_FR[ak.local_index(chi2_FR) == best_idx], axis=1)
    best_t = ak.firsts(t[ak.local_index(t) == best_idx], axis=1)

    good_chi2 = ((N_TOPS - n_selected_tops) > i) & (best_chi2 != FILL_VALUE)
    selected_b.append(ak.mask(best_t.b, good_chi2))
    selected_q1.append(ak.mask(best_t.w.j1, good_chi2)); selected_q2.append(ak.mask(best_t.w.j2, good_chi2))

    t_mask = t_mask & (
        ((t.w.j1["index"] != best_t.w.j1["index"]) 
            & (t.w.j1["index"] != best_t.w.j2["index"]) 
            & (t.w.j2["index"] != best_t.w.j1["index"]) 
            & (t.w.j2["index"] != best_t.w.j2["index"]) 
            & (t.b["index"] != best_t.b["index"]))
        | ~good_chi2
    )
    t_mask = ak.fill_none(t_mask, [], axis=0)

selected_b = ak.concatenate([selection[:, np.newaxis] for selection in selected_b], axis=1)
selected_q1 = ak.concatenate([selection[:, np.newaxis] for selection in selected_q1], axis=1)
selected_q2 = ak.concatenate([selection[:, np.newaxis] for selection in selected_q2], axis=1)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_q1, axis=1), axis=1)
print(f"n_selected after Fully-Resolved = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/FRt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_b, axis=1), axis=1) > i)
    dataset[f'TARGETS/FRt{i+1}/b'] = ak.fill_none(selected_b["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/q1'] = ak.fill_none(selected_q1["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/q2'] = ak.fill_none(selected_q2["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/assignment_probability'] = dataset[f'TARGETS/FRt{i+1}/detection_probability']


################################################
## Outputs ##
################################################

################################################
# Save out new h5 file
if SAVE_H5:
    out_filepath = os.path.join(DIRPATH, f"tt_hadronic_analysisEmu.h5")
    if os.path.exists(out_filepath): os.remove(out_filepath)
    with h5py.File(out_filepath, 'a') as f:
        with h5py.File(file_path, 'r') as test_f:
            for jet_class in test_f['INPUTS'].keys():
                for variable in test_f['INPUTS'][jet_class].keys():
                    if f'INPUTS/{jet_class}/{variable}' not in f:
                        f[f'INPUTS/{jet_class}/{variable}'] = test_f[f'INPUTS/{jet_class}/{variable}'][:]

        for label, column in dataset.items():
            # print(f"{label}, {ak.type(column)} - \n  {column}")
            f[label] = ak.to_numpy(column, allow_missing=False)

################################################
# Computes if χ² method found correct tops
def correct_mask(dataset, top_idx):
    return (
        (dataset[f'TARGETS/FBt{i+1}/bqq'] == tgt_FB_dict[f'FBt{top_idx}/bqq'])
        | (
            (dataset[f'TARGETS/SRqqt{i+1}/b'] == tgt_SRqq_dict[f'SRqqt{top_idx}/b'])
            & (dataset[f'TARGETS/SRqqt{i+1}/qq'] == tgt_SRqq_dict[f'SRqqt{top_idx}/qq'])
        ) | (
            (dataset[f'TARGETS/FRt{i+1}/b'] == tgt_FR_dict[f'FRt{top_idx}/b'])
            & (dataset[f'TARGETS/FRt{i+1}/q1'] == tgt_FR_dict[f'FRt{top_idx}/q1'])
            & (dataset[f'TARGETS/FRt{i+1}/q2'] == tgt_FR_dict[f'FRt{top_idx}/q2'])
        )
    )
