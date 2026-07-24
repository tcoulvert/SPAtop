import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import dp_to_TopNumProb, reset_collision_dp

N_AK4_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2

RESOLVED_CHI2_CUT = 45  # taken by-eye from boosted chi2 plots


def sel_pred_FRt_by_dp_ap(dps, aps, b_ps, q1_ps, q2_ps):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)

    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, TopNum)]

    # select the predicted b assignment via the indices
    b_ps_sel = b_ps[idx_sel]
    q1_ps_sel = q1_ps[idx_sel]
    q2_ps_sel = q2_ps[idx_sel]

    # require b, q1, q2 assignment are AK4 jet
    b_ak4_filter = b_ps_sel < N_AK4_JETS
    q1_ak4_filter = q1_ps_sel < N_AK4_JETS
    q2_ak4_filter = q2_ps_sel < N_AK4_JETS
    filter = b_ak4_filter & q1_ak4_filter & q2_ak4_filter

    b_ps_passed = b_ps_sel.mask[filter]
    b_ps_passed = ak.drop_none(b_ps_passed)

    q1_ps_passed = q1_ps_sel.mask[filter]
    q1_ps_passed = ak.drop_none(q1_ps_passed)

    q2_ps_passed = q2_ps_sel.mask[filter]
    q2_ps_passed = ak.drop_none(q2_ps_passed)

    return b_ps_passed, q1_ps_passed, q2_ps_passed

def sel_target_FRt_by_mask(b_ts, q1_ts, q2_ts, FRt_pts, FRt_masks):
    b_ts_selected = b_ts.mask[FRt_masks]
    b_ts_selected = ak.drop_none(b_ts_selected)

    q1_ts_selected = q1_ts.mask[FRt_masks]
    q1_ts_selected = ak.drop_none(q1_ts_selected)

    q2_ts_selected = q2_ts.mask[FRt_masks]
    q2_ts_selected = ak.drop_none(q2_ts_selected)

    FRt_selected_pts = FRt_pts.mask[FRt_masks]
    FRt_selected_pts = ak.drop_none(FRt_selected_pts)

    return b_ts_selected, q1_ts_selected, q2_ts_selected, FRt_selected_pts


# A pred look up table is in shape
# [event,
#    pred_FRt,
#       [correct_or_not, pred_pt, overlap_w_FRt_reco, has_boost_FBt_target, which_FRt_target]]
@nb.njit
def gen_pred_FRt_LUT(
    b_ps_passed, q1_ps_passed, q2_ps_passed, 
    b_ts_selected, q1_ts_selected, q2_ts_selected, 
    js,
    builder
):
    # for each event
    for b_ps_e, q1_ps_e, q2_ps_e, b_ts_e, q1_ts_e, q2_ts_e, jets_e in zip(
        b_ps_passed, q1_ps_passed, q2_ps_passed, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        js
    ):
        # for each predicted FRt assignment, check if any target t have a same FBt assignment
        builder.begin_list()

        for b_p, q1_p, q2_p in zip(b_ps_e, q1_ps_e, q2_ps_e):

            correct = 0
            predFRt_pt = (jets_e[b_p] + jets_e[q1_p] + jets_e[q2_p]).pt
            for b_t, q1_t, q2_t in zip(b_ts_e, q1_ts_e, q2_ts_e):
                if set((b_p, q1_p, q2_p)) == set((b_t, q1_t, q2_t)): correct = 1

            builder.begin_list()
            builder.append(correct)
            builder.append(predFRt_pt)
            builder.append(b_p)
            builder.append(q1_p)
            builder.append(q2_p)
            builder.end_list()

        builder.end_list()

    return builder


# A target look up table is in shape
# [event,
#    target_top,
#        target_FBt_assign,
#           [retrieved, targetFRt_pt, can_boost_reco]]
@nb.njit
def gen_target_FRt_LUT(
    b_ps_passed, q1_ps_passed, q2_ps_passed, 
    b_ts_selected, q1_ts_selected, q2_ts_selected, 
    FRt_pts, 
    builder
):
    # for each event
    for b_ps_e, q1_ps_e, q2_ps_e, b_ts_e, q1_ts_e, q2_ts_e, FRt_pts_e in zip(
        b_ps_passed, q1_ps_passed, q2_ps_passed, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        FRt_pts
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        for b_t, q1_t, q2_t, FRt_pt in zip(b_ts_e, q1_ts_e, q2_ts_e, FRt_pts_e):
            
            retrieved = 0
            for b_p, q1_p, q2_p in zip(b_ps_e, q1_ps_e, q2_ps_e):
                if set((b_p, q1_p, q2_p)) == set((b_t, q1_t, q2_t)): retrieved = 1

            builder.begin_list()
            builder.append(retrieved)
            builder.append(FRt_pt)
            builder.end_list()

        builder.end_list()

    return builder

# Calculate the most probable number of resonant particles

def parse_resolved_w_target(
    testfile, predfile, 
    chi2_cut=RESOLVED_CHI2_CUT
):
    if not any('FR' in key for key in predfile["TARGETS"].keys()): return None, None

    # FRt pt
    FRt1_pt = np.array(testfile["TARGETS"]["FRt1"]["pt"])
    FRt2_pt = np.array(testfile["TARGETS"]["FRt2"]["pt"])
    FRt_pts = np.concatenate((FRt1_pt.reshape(-1, 1), FRt2_pt.reshape(-1, 1)), axis=1)
    FRt_pts = ak.Array(FRt_pts)


    # resolved MASK
    FRt1_mask = np.array(testfile["TARGETS"]["FRt1"]["MASK"])
    FRt2_mask = np.array(testfile["TARGETS"]["FRt2"]["MASK"])
    FRt_masks = np.concatenate((FRt1_mask.reshape(-1, 1), FRt2_mask.reshape(-1, 1)), axis=1)
    FRt_masks = ak.Array(FRt_masks)


    # target jets
    b_FRt1_t = np.array(testfile["TARGETS"]["FRt1"]["b"])
    b_FRt2_t = np.array(testfile["TARGETS"]["FRt2"]["b"])

    q1_FRt1_t = np.array(testfile["TARGETS"]["FRt1"]["q1"])
    q1_FRt2_t = np.array(testfile["TARGETS"]["FRt2"]["q1"])

    q2_FRt1_t = np.array(testfile["TARGETS"]["FRt1"]["q2"])
    q2_FRt2_t = np.array(testfile["TARGETS"]["FRt2"]["q2"])

    b_ts = np.concatenate(
        (b_FRt1_t.reshape(-1, 1), b_FRt2_t.reshape(-1, 1)), axis=1
    )
    b_ts = ak.Array(b_ts)
    q1_ts = np.concatenate(
        (q1_FRt1_t.reshape(-1, 1), q1_FRt2_t.reshape(-1, 1)), axis=1
    )
    q1_ts = ak.Array(q1_ts)
    q2_ts = np.concatenate(
        (q2_FRt1_t.reshape(-1, 1), q2_FRt2_t.reshape(-1, 1)), axis=1
    )
    q2_ts = ak.Array(q2_ts)


    # pred jets
    b_FRt1_p = np.array(predfile["TARGETS"]["FRt1"]["b"])
    b_FRt2_p = np.array(predfile["TARGETS"]["FRt2"]["b"])

    q1_FRt1_p = np.array(predfile["TARGETS"]["FRt1"]["q1"])
    q1_FRt2_p = np.array(predfile["TARGETS"]["FRt2"]["q1"])

    q2_FRt1_p = np.array(predfile["TARGETS"]["FRt1"]["q2"])
    q2_FRt2_p = np.array(predfile["TARGETS"]["FRt2"]["q2"])

    b_ps = np.concatenate(
        (b_FRt1_p.reshape(-1, 1), b_FRt2_p.reshape(-1, 1)), axis=1
    )
    b_ps = ak.Array(b_ps)
    q1_ps = np.concatenate(
        (q1_FRt1_p.reshape(-1, 1), q1_FRt2_p.reshape(-1, 1)), axis=1
    )
    q1_ps = ak.Array(q1_ps)
    q2_ps = np.concatenate(
        (q2_FRt1_p.reshape(-1, 1), q2_FRt2_p.reshape(-1, 1)), axis=1
    )
    q2_ps = ak.Array(q2_ps)


    try:
        # jet detection probability
        dp_FRt1 = np.array(predfile["TARGETS"]["FRt1"]["detection_probability"])
        dp_FRt2 = np.array(predfile["TARGETS"]["FRt2"]["detection_probability"])
        # jet assignment probability
        ap_FRt1 = np.array(predfile["TARGETS"]["FRt1"]["assignment_probability"])
        ap_FRt2 = np.array(predfile["TARGETS"]["FRt2"]["assignment_probability"])
    except:
        # resolved top detection probability
        dp_FRt1 = np.logical_and(
            np.array(predfile["TARGETS"]["FRt1"]["MASK"]), np.array(predfile["TARGETS"]["FRt1"]["chi2"]) < chi2_cut
        ).astype("float")
        dp_FRt2 = np.logical_and(
            np.array(predfile["TARGETS"]["FRt2"]["MASK"]), np.array(predfile["TARGETS"]["FRt2"]["chi2"]) < chi2_cut
        ).astype("float")

        # jet assignment probability
        ap_FRt1 = np.logical_and(
            np.array(predfile["TARGETS"]["FRt1"]["MASK"]), np.array(predfile["TARGETS"]["FRt1"]["chi2"]) < chi2_cut
        ).astype("float")
        ap_FRt2 = np.logical_and(
            np.array(predfile["TARGETS"]["FRt2"]["MASK"]), np.array(predfile["TARGETS"]["FRt2"]["chi2"]) < chi2_cut
        ).astype("float")

    dps = np.concatenate((dp_FRt1.reshape(-1, 1), dp_FRt2.reshape(-1, 1)), axis=1)
    aps = np.concatenate((ap_FRt1.reshape(-1, 1), ap_FRt2.reshape(-1, 1)), axis=1)
    # convert some numpy arrays to ak arrays
    dps = reset_collision_dp(dps, aps)


    # reconstruct jet 4-momentum objects
    j_pt = np.array(testfile["INPUTS"]["Jets"]["pt"])
    j_eta = np.array(testfile["INPUTS"]["Jets"]["eta"])
    j_phi = np.array(testfile["INPUTS"]["Jets"]["phi"])
    j_mass = np.array(testfile["INPUTS"]["Jets"]["mass"])
    js = ak.zip(
        {
            "pt": j_pt,
            "eta": j_eta,
            "phi": j_phi,
            "mass": j_mass,
        },
        with_name="Momentum4D",
    )


    # select predictions and targets
    b_ts_selected, q1_ts_selected, q2_ts_selected, FRt_selected_pts = sel_target_FRt_by_mask(
        b_ts, q1_ts, q2_ts, FRt_pts, FRt_masks
    )
    b_ps_selected, q1_ps_selected, q2_ps_selected = sel_pred_FRt_by_dp_ap(dps, aps, b_ps, q1_ps, q2_ps)


    # generate look up tables
    LUT_pred = gen_pred_FRt_LUT(
        b_ps_selected, q1_ps_selected, q2_ps_selected, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        js, 
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_FRt_LUT(
        b_ps_selected, q1_ps_selected, q2_ps_selected,
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        FRt_selected_pts, 
        ak.ArrayBuilder(),
    ).snapshot()


    return LUT_pred, LUT_target
