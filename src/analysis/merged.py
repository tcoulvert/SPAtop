import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

# from src.analysis.utils import dp_to_TopNumProb, reset_collision_dp, nalpha
from src.analysis.utils import dp_to_TopNumProb, n_alpha

N_AK4_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2

# def get_unoverlapped_jet_index(fjs, js, dR_min=0.5):
#     overlapped = ak.sum(js[:, np.newaxis].deltaR(fjs) < dR_min, axis=-2) > 0
#     jet_index_passed = ak.local_index(js).mask[~overlapped]
#     jet_index_passed = ak.drop_none(jet_index_passed)
#     return jet_index_passed



def sel_target_t_by_mask(target_jets, target_pts, target_masks):
    selected_target_jets = target_jets.mask[target_masks]

    selected_target_pts = target_pts.mask[target_masks]

    return selected_target_jets, selected_target_pts

def sel_pred_t_by_dp_ap(predicted_jets, predicted_pts, dps, aps):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps, N_TOPS)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)

    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, TopNum)]

    # select the predicted q and qq assignment via the indices
    selected_predicted_jets = predicted_jets[idx_sel]
    selected_predicted_pts = predicted_pts[idx_sel]

    # selected jets assigned to jets
    filter = ak.all(~ak.is_none(selected_predicted_jets), axis=-1)
    selected_predicted_jets = selected_predicted_jets.mask[filter]
    selected_predicted_pts = selected_predicted_pts[filter]

    return selected_predicted_jets, selected_predicted_pts


# A pred look up table is in shape
# [event,
#    pred_SRt,
#       [correct_or_not, pred_pt, overlap_w_SRt_reco, has_boost_FBt_target, which_SRt_target]]
@nb.njit
def gen_pred_merged_LUT(
    q_ps_passed, qq_ps_passed,
    q_ts_selected, qq_ts_selected,
    js, goodJetIdx, 
    fjs, goodFatJetIdx, FBt_overlap_selected, 
    builder
):
    # for each event
    for q_ps_e, qq_ps_e, q_ts_e, qq_ts_e, jets_e, goodJetIdx_e, fatjets_e, goodFatJetIdx_e, FBt_overlap_e in zip(
        q_ps_passed, qq_ps_passed,
        q_ts_selected, qq_ts_selected,
        js, goodJetIdx, 
        fjs, goodFatJetIdx,
        FBt_overlap_selected
    ):
        # for each predicted FRt assignment, check if any target t have a same FBt assignment
        builder.begin_list()

        for q_p, qq_p in zip(q_ps_e, qq_ps_e):

            if (q_p in goodJetIdx_e) and (qq_p - N_AK4_JETS in goodFatJetIdx_e):
                overlap = 0
            else:
                overlap = 1
            correct = 0
            has_t_FBt = -1
            FBt = -1

            predFRt_pt = (jets_e[q_p] + fatjets_e[qq_p - N_AK4_JETS]).pt

            for i, (q_t, qq_t, FBt_overlap) in enumerate(zip(q_ts_e, qq_ts_e, FBt_overlap_e)):
                if set((q_p, qq_p - N_AK4_JETS)) == set((q_t, qq_t)):
                    correct = 1
                    has_t_FBt = FBt_overlap
                    FBt = i

            builder.begin_list()
            builder.append(correct)
            builder.append(predFRt_pt)
            builder.append(overlap)
            builder.append(has_t_FBt)
            builder.append(FBt)
            builder.append(q_p)
            builder.append(qq_p)
            builder.end_list()

        builder.end_list()

    return builder


# A target look up table is in shape
# [event,
#    target_top,
#        target_FBt_assign,
#           [retrieved, targetSRt_pt, can_boost_reco]]
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

        for jets1, toppt in zip(jets1_event, toppt_event):
            retrieved = 0
            for jets2 in jets2_event:
                for jet1, jet2 in zip(jets1, jets2):
                    if jet1["index"] == jet2["index"]: retrieved += 1

            builder.begin_list()
            builder.append(retrieved == len(jets1))
            builder.append(toppt)
            builder.end_list()

        builder.end_list()

    return builder


def parse_merged_w_target(
    testfile, predfile, reco_regex: str=''
):  
    reconstructions = [key for key in testfile["TARGETS"].keys() if reco_regex in key]
    jet_labels = {reco: list(testfile["TARGETS"][reco].keys()) for reco in reconstructions}

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
                    fatjets[ak.local_index(fatjets) == np.array(file["TARGETS"][reco][label]) 
                        | fatjets["index"] == np.array(file["TARGETS"][reco][label])]
                )[:, np.newaxis]
                for label in jet_labels[reco]
            ], axis=1).reshape(-1, 1, len(jet_labels[reco]))
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
    fatjets = ak.from_regular(ak.zip({
        "pt": np.array(testfile["INPUTS"]["BoostedJets"]["fj_pt"]),
        "eta": np.array(testfile["INPUTS"]["BoostedJets"]["fj_eta"]),
        "phi": np.array(testfile["INPUTS"]["BoostedJets"]["fj_phi"]),
        "mass": np.array(testfile["INPUTS"]["BoostedJets"]["fj_mass"])
    }, with_name="Momentum4D"))
    fatjets["index"] = ak.local_index(fatjets) + N_AK4_JETS

    # target pt
    target_pts = get_numerical(testfile, "pt")

    # target MASK
    target_masks = get_numerical(testfile, "MASK")

    # target jets
    target_jets = get_jets(testfile)

    # predicted jets
    predicted_jets = get_jets(predfile)
    predicted_pts = ak.sum(predicted_jets, axis=-1).pt

    # predicted probabilities
    dps = get_numerical(predfile, "detection_probability")
    aps = get_numerical(predfile, "assignment_probability")
    # # convert some numpy arrays to ak arrays
    # dps = reset_collision_dp(dps, aps)



    # select predictions and targets
    selected_target_jets, selected_target_pts = sel_target_t_by_mask(target_jets, target_pts, target_masks)
    selected_predicted_jets, selected_predicted_pts = sel_pred_t_by_dp_ap(predicted_jets, predicted_pts, dps, aps)


    # generate look up tables
    LUT_pred = generate_LUT(
        selected_predicted_jets, selected_target_jets,
        selected_predicted_pts,
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_SRt_LUT(
        q_ps_selected, qq_ps_selected,
        q_ts_selected, qq_ts_selected,
        SRt_selected_pts, 
        overlap_selected,
        ak.ArrayBuilder(),
    ).snapshot()


    return LUT_pred, LUT_target
