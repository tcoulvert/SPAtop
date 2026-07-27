import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()
import time

from src.analysis.utils import best_reco_order, reset_collision_dp, dp_to_TopNumProb, n_alpha

N_AK5_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2

def sel_target_t_by_mask(target_jets, target_pts, target_masks, dps, aps):
    # get the best (dp x ap) jet assignment indices
    idx_descend = best_reco_order(dps, aps)
    idx_sel = [idx_e for idx_e in idx_descend]

    selected_target_jets = target_jets[idx_sel]
    selected_target_pts = target_pts[idx_sel]

    filter = (ak.all(~ak.is_none(selected_target_jets, axis=-1), axis=-1) & target_masks)
    selected_target_jets = ak.mask(selected_target_jets, filter)
    selected_target_pts = ak.where(filter, selected_target_pts, -999)

    return selected_target_jets, selected_target_pts

def sel_pred_t_by_dp_ap(predicted_jets, predicted_pts, dps, aps):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps, N_TOPS)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the best N (dp x ap) jet assignment indices
    idx_descend = best_reco_order(dps, aps)
    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, TopNum)]

    selected_predicted_jets = predicted_jets[idx_sel]
    selected_predicted_pts = predicted_pts[idx_sel]

    # selected jets assigned to jets
    filter = ak.all(~ak.is_none(selected_predicted_jets, axis=-1), axis=-1)
    selected_predicted_jets = ak.mask(selected_predicted_jets, filter)
    selected_predicted_pts = ak.where(filter, selected_predicted_pts, -999)

    return selected_predicted_jets, selected_predicted_pts


# A look up table is in shape
# [event,
#    valid_jets1,
#        [retrieved, pt]]
@nb.njit
def generate_LUT(
    selected_jets1, selected_jets2,
    selected_toppt,
    builder
):
    # for each event
    for jets1_event, jets2_event, toppt_event in zip(
        selected_jets1, selected_jets2,
        selected_toppt
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        matched_idxs = set()
        for jets1, toppt in zip(jets1_event, toppt_event):
            if jets1 is None: continue

            retrieved = 0
            for i, jets2 in enumerate(jets2_event):
                if jets2 is None: continue
                if i in matched_idxs: continue

                n_jet1s = 0; n_matched = 0
                for jet1 in jets1:
                    n_jet1s += 1
                    for jet2 in jets2:
                        if (
                            jet1.pt == jet2.pt and jet1.eta == jet2.eta 
                            and jet1.phi == jet2.phi and jet1.mass == jet2.mass
                        ): n_matched += 1

                if n_matched == n_jet1s: retrieved = 1; matched_idxs.add(i); break

            builder.begin_list()
            builder.append(retrieved)
            builder.append(toppt)
            builder.end_list()

        builder.end_list()

    return builder


def parse_merged_w_target(
    testfile, predfile, reco_regex: str=''
):  
    print(f"Processing reco: {reco_regex}")
    reconstructions = [key for key in testfile["TARGETS"].keys() if reco_regex in key]
    N_TOPS = np.max([int(k) for key in reconstructions for k in key if k.isdigit()])
    print(f"Number of tops: {N_TOPS}")
    jet_labels = {reco: [key for key in predfile["TARGETS"][reco].keys() if 'prob' not in key] for reco in reconstructions}

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
    # jets["index"] = ak.local_index(jets)
    jets = ak.with_field(jets, ak.local_index(jets, axis=1), "index")
    N_AK5_JETS = ak.max(ak.local_index(jets), axis=None) + 1
    print(f"Number of AK5 jets: {N_AK5_JETS}")
    fatjets = ak.from_regular(ak.zip({
        "pt": np.array(testfile["INPUTS"]["BoostedJets"]["fj_pt"]),
        "eta": np.array(testfile["INPUTS"]["BoostedJets"]["fj_eta"]),
        "phi": np.array(testfile["INPUTS"]["BoostedJets"]["fj_phi"]),
        "mass": np.array(testfile["INPUTS"]["BoostedJets"]["fj_mass"])
    }, with_name="Momentum4D"))
    # fatjets["index"] = ak.local_index(fatjets) + N_AK5_JETS
    fatjets = ak.with_field(fatjets, ak.local_index(fatjets, axis=1) + N_AK5_JETS, "index")
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
    dps = reset_collision_dp(dps, aps)


    # select predictions and targets
    selected_target_jets, selected_target_pts = sel_target_t_by_mask(target_jets, target_pts, target_masks, dps, aps)
    selected_predicted_jets, selected_predicted_pts = sel_pred_t_by_dp_ap(predicted_jets, predicted_pts, dps, aps)


    # generate look up tables
    LUT_pred = generate_LUT(
        selected_predicted_jets, selected_target_jets,
        selected_predicted_pts,
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = generate_LUT(
        selected_target_jets, selected_predicted_jets,
        selected_target_pts,
        ak.ArrayBuilder()
    ).snapshot()


    return LUT_pred, LUT_target
