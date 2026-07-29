import itertools

import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import reco_reorder, reset_collision_dp, dp_to_TopNumProb, match_jet, get_symmetries, n_alpha

N_AK5_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2
DELTARS = None
SYMMETRIES = None

def sel_target_t_by_mask(target_jets, target_pts, target_masks):
    filter = target_masks
    selected_target_jets = ak.mask(target_jets, filter)
    selected_target_pts = ak.where(filter, target_pts, -999)

    return selected_target_jets, selected_target_pts

def sel_pred_t_by_prob(predicted_jets, predicted_pts, dps, aps):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the best N (dp x ap) jet assignment indices
    idx_sel = reco_reorder(predicted_jets, dps, aps, TopNum, N_TOPS, DELTARS)

    # selected jets assigned to jets
    filter = ak.all(~ak.is_none(predicted_jets, axis=-1), axis=-1)
    selected_predicted_jets = ak.mask(predicted_jets, filter)
    selected_predicted_pts = ak.where(filter, predicted_pts, -999)

    return selected_predicted_jets, selected_predicted_pts, idx_sel


# A look up table is in shape
# [event x valid_predjets,
#        [retrieved, pt]]
def generate_pred_LUT(predicted_jets, target_jets, predicted_toppt, selected_order):
    return generate_one_pred_LUT(
        predicted_jets, target_jets, predicted_toppt,
        selected_order, SYMMETRIES,
        ak.ArrayBuilder()
    ).snapshot()

@nb.njit
def generate_one_pred_LUT(
    predicted_jets, target_jets, predicted_toppt,
    selected_order, symmetries,
    builder
):
    # for each event
    for pjets_event, tjets_event, toppt_event, order_event in zip(
        predicted_jets, target_jets,
        predicted_toppt,
        selected_order
    ):
        # for each prediction per event, in order of best probs
        for pred_idx in order_event:
            pjets, toppt = pjets_event[pred_idx], toppt_event[pred_idx]
            if pjets is None: continue

            retrieved = 0
            # check all targets of matching reco (i.e. account for symmetry of top label exchange)
            target_idxs = [pred_idx - i for i in range(1, (pred_idx % N_TOPS)+1)][::-1]+[pred_idx]+[pred_idx + i for i in range(1, N_TOPS-(pred_idx % N_TOPS))]
            for targ_idx in target_idxs:
                tjets = tjets_event[targ_idx]
                if tjets is None: continue

                # check all valid labels (i.e. account for symmetry of jet labels)
                for symand in symmetries[pred_idx]:
                    n_matched = 0
                    for labels in symand:
                        if match_jet(pjets[labels[0]], tjets[labels[1]]): n_matched += 1
                    if n_matched == len(symand): retrieved = 1; break

                if retrieved: break

            builder.begin_list()
            builder.append(retrieved)
            builder.append(toppt)
            builder.end_list()

    return builder


# A look up table is in shape
# [event x valid_targjets,
#        [retrieved, pt]]
def generate_target_LUT(
    target_jets, predicted_jets, target_toppt, selected_order
):
    return ak.concatenate([
        generate_one_target_LUT(
            target_jets[:, i::N_TOPS], predicted_jets, target_toppt[:, i::N_TOPS], selected_order, SYMMETRIES, ak.ArrayBuilder()
        ).snapshot() for i in range(N_TOPS)
    ], axis=0)

@nb.njit
def generate_one_target_LUT(
    target_jets, predicted_jets, target_toppt,
    selected_order, symmetries,
    builder
):
    # for each event
    for tjets_event, pjets_event, toppt_event, order_event in zip(
        target_jets, predicted_jets,
        target_toppt,
        selected_order
    ):
        # Check for any valid targets
        toppt = -999.0
        for tjets, toppt_ in zip(tjets_event, toppt_event):
            if tjets is None: continue
            toppt = toppt_; break
        if toppt < 0: continue
        retrieved = 0

        # Try all preds to find 1 match
        for pred_idx in order_event:
            pjets = pjets_event[pred_idx]
            if pjets is None: continue

            targ_idx = pred_idx // N_TOPS
            tjets = tjets_event[targ_idx]
            if tjets is None: continue

            # check all valid labels (i.e. account for symmetry of jet labels)
            for symand in symmetries[pred_idx]:
                n_matched = 0
                for labels in symand:
                    if match_jet(pjets[labels[0]], tjets[labels[1]]): n_matched += 1
                if n_matched == len(symand): retrieved = 1; break

            if retrieved: break

        builder.begin_list()
        builder.append(retrieved)
        builder.append(toppt)
        builder.end_list()

    return builder


def parse_merged_w_target(
    testfile, predfile, reco_regex: str='', chi2: bool=False
):  
    print(f"Processing reco: {reco_regex}")
    global N_TOPS, N_AK5_JETS, N_AK8_JETS, DELTARS, SYMMETRIES
    reconstructions = sorted([key for key in predfile["TARGETS"].keys() if reco_regex in key])
    if len(reconstructions) == 0: return None, None
    N_TOPS = np.max([int(k) for key in reconstructions for k in key if k.isdigit()])
    print(f"Number of tops: {N_TOPS}")
    jet_labels = {reco: sorted([key for key in predfile["TARGETS"][reco].keys() if 'prob' not in key]) for reco in reconstructions}
    DELTARS = [[0.8 if n_alpha(label) > 1 else 0.5 for label in jet_labels[reco]] for reco in reconstructions]
    SYMMETRIES = get_symmetries(reconstructions, jet_labels)

    def get_numerical(file, key: str):
        return ak.Array(
            np.concatenate([
                np.array(file["TARGETS"][reco][key]).reshape(-1, 1)
                for reco in reconstructions
            ], axis=1)
        )
    def get_jets(file):
        return ak.concatenate([
            ak.concatenate([
                ak.firsts(
                    jets[ak.local_index(jets) == np.array(file["TARGETS"][reco][label])]
                    if n_alpha(label) == 1 else
                    fatjets[(ak.local_index(fatjets) == np.array(file["TARGETS"][reco][label])) 
                        | (fatjets["index"] == np.array(file["TARGETS"][reco][label]))]
                )[:, np.newaxis]
                for label in jet_labels[reco]
            ], axis=1)[:, np.newaxis, :]
            for reco in reconstructions
        ], axis=1)

    # jet 4-momentums
    jets = ak.from_regular(ak.zip({
        "pt": np.array(testfile["INPUTS"]["Jets"]["pt"]),
        "eta": np.array(testfile["INPUTS"]["Jets"]["eta"]),
        "phi": np.array(testfile["INPUTS"]["Jets"]["phi"]),
        "mass": np.array(testfile["INPUTS"]["Jets"]["mass"])
    },  with_name="Momentum4D"))
    jets["index"] = ak.local_index(jets)
    N_AK5_JETS = ak.max(ak.local_index(jets), axis=None) + 1
    print(f"Number of AK5 jets: {N_AK5_JETS}")
    fatjets = ak.from_regular(ak.zip({
        "pt": np.array(testfile["INPUTS"]["BoostedJets"]["fj_pt"]),
        "eta": np.array(testfile["INPUTS"]["BoostedJets"]["fj_eta"]),
        "phi": np.array(testfile["INPUTS"]["BoostedJets"]["fj_phi"]),
        "mass": np.array(testfile["INPUTS"]["BoostedJets"]["fj_mass"])
    }, with_name="Momentum4D"))
    fatjets["index"] = ak.local_index(fatjets) + N_AK5_JETS
    N_AK8_JETS = ak.max(ak.local_index(fatjets), axis=None) + 1
    print(f"Number of AK8 jets: {N_AK8_JETS}")

    # target pt
    target_pts = get_numerical(testfile, "pt")

    # target MASK
    target_masks = get_numerical(testfile, "MASK")

    # target jets
    target_jets = get_jets(testfile)

    # predicted jets
    predicted_jets = get_jets(predfile)
    predicted_pts = ak.Array(ak.sum(predicted_jets, axis=-1).pt)

    # predicted probabilities
    dps = get_numerical(predfile, "detection_probability")
    aps = get_numerical(predfile, "assignment_probability")
    if not chi2: dps = reset_collision_dp(dps, aps)


    # select predictions and targets
    selected_target_jets, selected_target_pts = sel_target_t_by_mask(target_jets, target_pts, target_masks)
    selected_predicted_jets, selected_predicted_pts, selected_order = sel_pred_t_by_prob(predicted_jets, predicted_pts, dps, aps)

    # generate look up tables
    LUT_pred = generate_pred_LUT(selected_predicted_jets, selected_target_jets, selected_predicted_pts, selected_order)
    LUT_target = generate_target_LUT(selected_target_jets, selected_predicted_jets, selected_target_pts, selected_order)

    return LUT_pred, LUT_target
