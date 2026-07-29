import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import reco_reorder, reset_collision_dp, dp_to_TopNumProb, n_alpha

N_AK5_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2

def sel_target_t_by_mask(target_jets, target_pts, target_masks):
    filter = target_masks
    selected_target_jets = ak.mask(target_jets, filter)
    selected_target_pts = ak.where(filter, target_pts, -999)

    return selected_target_jets, selected_target_pts

def sel_pred_t_by_prob(predicted_jets, predicted_pts, dps, aps, deltaRs):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the best N (dp x ap) jet assignment indices
    idx_sel = reco_reorder(predicted_jets, dps, aps, TopNum, N_TOPS, deltaRs)
    selected_predicted_jets = predicted_jets[idx_sel]
    selected_predicted_pts = predicted_pts[idx_sel]

    # selected jets assigned to jets
    filter = ak.all(~ak.is_none(selected_predicted_jets, axis=-1), axis=-1)
    selected_predicted_jets = ak.mask(selected_predicted_jets, filter)
    selected_predicted_pts = ak.where(filter, selected_predicted_pts, -999)

    return selected_predicted_jets, selected_predicted_pts


# A look up table is in shape
# [event x valid_predjets,
#        [retrieved, pt]]
def generate_pred_LUT(predicted_jets, target_jets, predicted_toppt):
    return generate_one_pred_LUT(
        predicted_jets, target_jets, predicted_toppt,
        ak.ArrayBuilder()
    ).snapshot()

@nb.njit
def generate_one_pred_LUT(
    predicted_jets, target_jets, predicted_toppt,
    builder
):
    # for each event
    for pjets_event, tjets_event, toppt_event in zip(
        predicted_jets, target_jets,
        predicted_toppt
    ):
        matched_idxs = set()
        for pjets, toppt in zip(pjets_event, toppt_event):
            if pjets is None: continue

            retrieved = 0
            for i, tjets in enumerate(tjets_event):
                if tjets is None: continue
                if i in matched_idxs: continue

                n_pjets = 0; n_matched = 0
                for pjet in pjets:
                    n_pjets += 1
                    for tjet in tjets:
                        if (
                            pjet.pt == tjet.pt and pjet.eta == tjet.eta 
                            and pjet.phi == tjet.phi and pjet.mass == tjet.mass
                        ): n_matched += 1

                if n_matched == n_pjets: retrieved = 1; matched_idxs.add(i); break

            builder.begin_list()
            builder.append(retrieved)
            builder.append(toppt)
            builder.end_list()

    return builder

# A look up table is in shape
# [event x valid_targjets,
#        [retrieved, pt]]
def generate_target_LUT(
    target_jets, predicted_jets, target_toppt,
):
    return ak.concatenate([
        generate_one_target_LUT(
            target_jets[:, i::N_TOPS], predicted_jets, target_toppt[:, i::N_TOPS], ak.ArrayBuilder()
        ).snapshot() for i in range(N_TOPS)
    ], axis=0)

@nb.njit
def generate_one_target_LUT(
    target_jets, predicted_jets, target_toppt,
    builder
):
    # for each event
    for tjets_event, pjets_event, toppt_event in zip(
        target_jets, predicted_jets,
        target_toppt
    ):
        retrieved = 0; toppt = -999.0
        for tjets, toppt_ in zip(tjets_event, toppt_event):
            if tjets is None: continue
            toppt = toppt_

            for pjets in pjets_event:
                if pjets is None: continue

                n_tjets = 0; n_matched = 0
                for tjet in tjets:
                    n_tjets += 1
                    for pjet in pjets:
                        if (
                            tjet.pt == pjet.pt and tjet.eta == pjet.eta 
                            and tjet.phi == pjet.phi and tjet.mass == pjet.mass
                        ): n_matched += 1

                if n_matched == n_tjets: retrieved = 1; break

        if toppt > 0:
            builder.begin_list()
            builder.append(retrieved)
            builder.append(toppt)
            builder.end_list()

    return builder


def parse_merged_w_target(
    testfile, predfile, reco_regex: str=''
):  
    print(f"Processing reco: {reco_regex}")
    reconstructions = sorted([key for key in predfile["TARGETS"].keys() if reco_regex in key])
    if len(reconstructions) == 0: return None, None
    N_TOPS = np.max([int(k) for key in reconstructions for k in key if k.isdigit()])
    print(f"Number of tops: {N_TOPS}")
    jet_labels = {reco: [key for key in predfile["TARGETS"][reco].keys() if 'prob' not in key] for reco in reconstructions}
    deltaRs = [[0.8 if n_alpha(label) > 1 else 0.5 for label in jet_labels[reco]] for reco in reconstructions]

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
    dps = reset_collision_dp(dps, aps)


    # select predictions and targets
    selected_target_jets, selected_target_pts = sel_target_t_by_mask(target_jets, target_pts, target_masks)
    selected_predicted_jets, selected_predicted_pts = sel_pred_t_by_prob(predicted_jets, predicted_pts, dps, aps, deltaRs)


    # generate look up tables
    LUT_pred = generate_pred_LUT(selected_predicted_jets, selected_target_jets, selected_predicted_pts)
    LUT_target = generate_target_LUT(selected_target_jets, selected_predicted_jets, selected_target_pts)

    return LUT_pred, LUT_target
