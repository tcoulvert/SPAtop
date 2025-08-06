import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import dp_to_TopNumProb, reset_collision_dp

N_AK4_JETS = 10
N_AK8_JETS = 2
N_AK15_JETS = 2
N_TOPS = 2

BOOSTED_CHI2_CUT = 45  # taken by-eye from boosted chi2 plots


def sel_pred_FBt_by_dp_ap(dps, aps, bqq_ps):
    # get most possible number of H_reco by dps
    TopNumProb = dp_to_TopNumProb(dps)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)
    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, TopNum)]

    # select the predicted bqq assignment via the indices
    bqq_ps_sel = bqq_ps[idx_sel]

    # require bqq assignment is a AK15 jet
    ak15Filter = bqq_ps_sel >= (N_AK4_JETS + N_AK8_JETS)
    bqq_ps_passed = bqq_ps_sel.mask[ak15Filter]
    bqq_ps_passed = ak.drop_none(bqq_ps_passed)

    return bqq_ps_passed


def sel_target_FBt_by_mask(bqq_ts, FBt_pts, FBt_masks):
    bqq_ts_selected = bqq_ts.mask[FBt_masks]
    bqq_ts_selected = ak.drop_none(bqq_ts_selected)

    FBt_selected_pts = FBt_pts.mask[FBt_masks]
    FBt_selected_pts = ak.drop_none(FBt_selected_pts)

    return bqq_ts_selected, FBt_selected_pts


# A pred look up table is in shape
# [event,
#    pred_FBt,
#       [correct, pred_FBt_pt]]
@nb.njit
def gen_pred_FBt_LUT(bqq_ps_passed, bqq_ts_selected, fj_pts, builder):
    # for each event
    for bqq_t_event, bqq_p_event, fj_pt_event in zip(bqq_ts_selected, bqq_ps_passed, fj_pts):
        # for each predicted bb assignment, check if any target H have a same bb assignment
        builder.begin_list()

        for i, bqq_p in enumerate(bqq_p_event):

            correct = 0.
            predFBt_pt = fj_pt_event[bqq_p - (N_AK4_JETS + N_AK8_JETS)]
            for bqq_t in bqq_t_event:
                if bqq_p - (N_AK4_JETS + N_AK8_JETS) == bqq_t:
                    correct = 1.

            builder.begin_list()
            builder.append(correct)
            builder.append(predFBt_pt)
            builder.end_list()

        builder.end_list()

    return builder


# A target look up table is in shape
# [event,
#    target_FBt,
#        target_bqq_assign,
#           [retrieved, targetFBt_pt]]
@nb.njit
def gen_target_FBt_LUT(bqq_ps_passed, bqq_ts_selected, targetFBt_pts, builder):
    # for each event
    for bqq_t_event, bqq_p_event, targetH_pts_event in zip(bqq_ts_selected, bqq_ps_passed, targetFBt_pts):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        for i, bqq_t in enumerate(bqq_t_event):

            retrieved = 0.
            targetFBt_pt = targetH_pts_event[i]
            for bqq_p in bqq_p_event:
                if bqq_p - (N_AK4_JETS + N_AK8_JETS) == bqq_t:
                    retrieved = 1.

            builder.begin_list()
            builder.append(retrieved)
            builder.append(targetFBt_pt)
            builder.end_list()
            
        builder.end_list()

    return builder


# generate pred/target LUT
# each entry corresponds to [recoFBt correct or not, reco FBt pt]
# or
# [targetFBt retrieved or not, target FBt pt]
def parse_boosted_w_target(
    testfile, predfile,
):
    # Collect H pt, mask, target and predicted jet and fjets for 3 Hs in each event
    # t pt
    FBt1_pt = np.array(testfile["TARGETS"]["FBt1"]["pt"])
    FBt2_pt = np.array(testfile["TARGETS"]["FBt2"]["pt"])
    FBt_pts = np.concatenate((FBt1_pt.reshape(-1, 1), FBt2_pt.reshape(-1, 1)), axis=1)
    FBt_pts = ak.Array(FBt_pts)

    # mask
    FBt1_mask = np.array(testfile["TARGETS"]["FBt1"]["mask"])
    FBt2_mask = np.array(testfile["TARGETS"]["FBt2"]["mask"])
    FBt_masks = np.concatenate((FBt1_mask.reshape(-1, 1), FBt2_mask.reshape(-1, 1)), axis=1)
    FBt_masks = ak.Array(FBt_masks)

    # target jet/fjets
    bqq_FBt1_t = np.array(testfile["TARGETS"]["FBt1"]["bqq"])
    bqq_FBt2_t = np.array(testfile["TARGETS"]["FBt2"]["bqq"])
    bqq_ts = np.concatenate(
        (bqq_FBt1_t.reshape(-1, 1), bqq_FBt2_t.reshape(-1, 1)), axis=1
    )
    bqq_ts = ak.Array(bqq_ts)

    # pred jet/fjets
    try:
        # pred assignment
        bqq_FBt1_p = np.array(predfile["TARGETS"]["FBt1"]["bqq"])
        bqq_FBt2_p = np.array(predfile["TARGETS"]["FBt2"]["bqq"])

        # FBt detection probability
        dp_FBt1 = np.array(predfile["TARGETS"]["FBt1"]["detection_probability"])
        dp_FBt2 = np.array(predfile["TARGETS"]["FBt2"]["detection_probability"])

        # FBt assignment probability
        ap_FBt1 = np.array(predfile["TARGETS"]["FBt1"]["assignment_probability"])
        ap_FBt2 = np.array(predfile["TARGETS"]["FBt2"]["assignment_probability"])
    except:
        # pred assignment
        bqq_FBt1_p = np.array(predfile["TARGETS"]["FBt1"]["bqq"]) + (N_AK4_JETS + N_AK8_JETS)
        bqq_FBt2_p = np.array(predfile["TARGETS"]["FBt2"]["bqq"]) + (N_AK4_JETS + N_AK8_JETS)

        # boosted top detection probability
        dp_FBt1 = np.array(
            np.logical_and(
                predfile["TARGETS"]["FBt1"]["mask"],
                predfile["TARGETS"]["FBt1"]["chi2"] < BOOSTED_CHI2_CUT
            )
        ).astype("float")
        dp_FBt2 = np.array(
            np.logical_and(
                predfile["TARGETS"]["FBt2"]["mask"],
                predfile["TARGETS"]["FBt2"]["chi2"] < BOOSTED_CHI2_CUT
            )
        ).astype("float")

        # veryfatjet assignment probability
        ap_FBt1 = np.array(
            np.logical_and(
                predfile["TARGETS"]["FBt1"]["mask"],
                predfile["TARGETS"]["FBt1"]["chi2"] < BOOSTED_CHI2_CUT
            )
        ).astype("float")
        ap_FBt2 = np.array(
            np.logical_and(
                predfile["TARGETS"]["FBt2"]["mask"],
                predfile["TARGETS"]["FBt2"]["chi2"] < BOOSTED_CHI2_CUT
            )
        ).astype("float")

    bqq_ps = np.concatenate((bqq_FBt1_p.reshape(-1, 1), bqq_FBt2_p.reshape(-1, 1)), axis=1)
    bqq_ps = ak.Array(bqq_ps)
    dps = np.concatenate((dp_FBt1.reshape(-1, 1), dp_FBt2.reshape(-1, 1)), axis=1)
    aps = np.concatenate((ap_FBt1.reshape(-1, 1), ap_FBt2.reshape(-1, 1)), axis=1)


    # collect veryfatjet kinematics
    vfj_pt = np.array(testfile["INPUTS"]["VeryBoostedJets"]["vfj_pt"])
    vfj_eta = np.array(testfile["INPUTS"]["VeryBoostedJets"]["vfj_eta"])
    vfj_phi = np.array(testfile["INPUTS"]["VeryBoostedJets"]["vfj_pt"])
    vfj_mass = np.array(testfile["INPUTS"]["VeryBoostedJets"]["vfj_pt"])
    vfjs = ak.zip({
        "pt": vfj_pt,
        "eta": vfj_eta,
        "phi": vfj_phi,
        "mass": vfj_mass,
    }, with_name="Momentum4D")


    # select predictions and targets
    bqq_ps_selected = sel_pred_FBt_by_dp_ap(dps, aps, bqq_ps)
    bqq_ts_selected, targetFBt_selected_pts = sel_target_FBt_by_mask(bqq_ts, FBt_pts, FBt_masks)


    # generate correct/retrieved LUT for pred/target respectively
    LUT_pred = gen_pred_FBt_LUT(
        bqq_ps_selected, bqq_ts_selected, vfjs.pt,
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_FBt_LUT(
        bqq_ps_selected, bqq_ts_selected, targetFBt_selected_pts,
        ak.ArrayBuilder()
    ).snapshot()


    # reconstruct FBt to remove overlapped ak4 & ak8 jets
    vfj_reco = vfjs[bqq_ps_selected - (N_AK4_JETS + N_AK8_JETS)]


    return LUT_pred, LUT_target, vfj_reco
