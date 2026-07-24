import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import dp_to_TopNumProb, reset_collision_dp

N_AK4_JETS = 10
N_AK8_JETS = 2
N_TOPS = 2



def sel_pred_SRt_by_dp_ap(dps, aps, q_ps, qq_ps):
    # get most possible number of Top_reco by dps
    TopNumProb = dp_to_TopNumProb(dps)
    TopNum = np.argmax(TopNumProb, axis=-1)

    # get the top N (dp x ap) jet assignment indices
    ps = dps * aps
    idx_descend = np.flip(np.argsort(ps, axis=-1), axis=-1)

    idx_sel = [idx_e[:N_e] for idx_e, N_e in zip(idx_descend, TopNum)]

    # select the predicted q and qq assignment via the indices
    q_ps_sel = q_ps[idx_sel]
    qq_ps_sel = qq_ps[idx_sel]

    # require q assignment is AK4 jet & qq assignment is AK8 jet
    q_ak4_filter = q_ps_sel < N_AK4_JETS
    qq_ak8_filter = (qq_ps_sel >= N_AK4_JETS) & ( qq_ps_sel < (N_AK4_JETS + N_AK8_JETS) )
    filter = q_ak4_filter & qq_ak8_filter

    q_ps_passed = q_ps_sel.mask[filter]
    q_ps_passed = ak.drop_none(q_ps_passed)

    qq_ps_passed = qq_ps_sel.mask[filter]
    qq_ps_passed = ak.drop_none(qq_ps_passed)

    return q_ps_passed, qq_ps_passed

def sel_target_SRt_by_mask(q_ts, qq_ts, SRt_pts, SRt_masks):
    q_ts_selected = q_ts.mask[SRt_masks]
    q_ts_selected = ak.drop_none(q_ts_selected)

    qq_ts_selected = qq_ts.mask[SRt_masks]
    qq_ts_selected = ak.drop_none(qq_ts_selected)

    SRt_selected_pts = SRt_pts.mask[SRt_masks]
    SRt_selected_pts = ak.drop_none(SRt_selected_pts)

    return q_ts_selected, qq_ts_selected, SRt_selected_pts


# A pred look up table is in shape
# [event,
#    pred_SRt,
#       [correct_or_not, pred_pt, overlap_w_SRt_reco, has_boost_FBt_target, which_SRt_target]]
@nb.njit
def gen_pred_SRt_LUT(
    q_ps_passed, qq_ps_passed,
    q_ts_selected, qq_ts_selected,
    js, fjs, 
    builder
):
    # for each event
    for q_ps_e, qq_ps_e, q_ts_e, qq_ts_e, jets_e, fatjets_e in zip(
        q_ps_passed, qq_ps_passed,
        q_ts_selected, qq_ts_selected,
        js, fjs,
    ):
        # for each predicted FRt assignment, check if any target t have a same FBt assignment
        builder.begin_list()

        for q_p, qq_p in zip(q_ps_e, qq_ps_e):

            correct = 0
            predFRt_pt = (jets_e[q_p] + fatjets_e[qq_p - N_AK4_JETS]).pt
            for q_t, qq_t in zip(q_ts_e, qq_ts_e):
                if set((q_p, qq_p - N_AK4_JETS)) == set((q_t, qq_t)): correct = 1

            builder.begin_list()
            builder.append(correct)
            builder.append(predFRt_pt)
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
def gen_target_SRt_LUT(
    q_ps_passed, qq_ps_passed,
    q_ts_selected, qq_ts_selected,
    SRt_pts, 
    builder
):
    # for each event
    for q_ps_e, qq_ps_e, q_ts_e, qq_ts_e, SRt_pts_e in zip(
        q_ps_passed, qq_ps_passed,
        q_ts_selected, qq_ts_selected,
        SRt_pts
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        for q_t, qq_t, SRt_pt in zip(q_ts_e, qq_ts_e, SRt_pts_e):
            
            retrieved = 0
            for q_p, qq_p in zip(q_ps_e, qq_ps_e):
                if set((q_p, qq_p - N_AK4_JETS)) == set((q_t, qq_t)): retrieved = 1

            builder.begin_list()
            builder.append(retrieved)
            builder.append(SRt_pt)
            builder.end_list()

        builder.end_list()

    return builder


def parse_semi_resolved_w_target(
    testfile, predfile, method,
):
    if not any(f'SR{method}' in key for key in predfile["TARGETS"].keys()): return None, None

    # FRt pt
    SRt1_pt = np.array(testfile["TARGETS"][f"SR{method}t1"]["pt"])
    SRt2_pt = np.array(testfile["TARGETS"][f"SR{method}t2"]["pt"])
    SRt_pts = np.concatenate((SRt1_pt.reshape(-1, 1), SRt2_pt.reshape(-1, 1)), axis=1)
    SRt_pts = ak.Array(SRt_pts)


    # resolved MASK
    SRt1_mask = np.array(testfile["TARGETS"][f"SR{method}t1"]["MASK"])
    SRt2_mask = np.array(testfile["TARGETS"][f"SR{method}t2"]["MASK"])
    SRt_masks = np.concatenate((SRt1_mask.reshape(-1, 1), SRt2_mask.reshape(-1, 1)), axis=1)
    SRt_masks = ak.Array(SRt_masks)


    # target jets
    q_SRt1_t = np.array(testfile["TARGETS"][f"SR{method}t1"][f"{'b' if method == 'qq' else 'q'}"])
    q_SRt2_t = np.array(testfile["TARGETS"][f"SR{method}t2"][f"{'b' if method == 'qq' else 'q'}"])

    qq_SRt1_t = np.array(testfile["TARGETS"][f"SR{method}t1"][f"{method}"])
    qq_SRt2_t = np.array(testfile["TARGETS"][f"SR{method}t2"][f"{method}"])

    q_ts = np.concatenate(
        (q_SRt1_t.reshape(-1, 1), q_SRt2_t.reshape(-1, 1)), axis=1
    )
    q_ts = ak.Array(q_ts)
    qq_ts = np.concatenate(
        (qq_SRt1_t.reshape(-1, 1), qq_SRt2_t.reshape(-1, 1)), axis=1
    )
    qq_ts = ak.Array(qq_ts)


    # pred jets
    q_SRt1_p = np.array(predfile["TARGETS"][f"SR{method}t1"][f"{'b' if method == 'qq' else 'q'}"])
    q_SRt2_p = np.array(predfile["TARGETS"][f"SR{method}t2"][f"{'b' if method == 'qq' else 'q'}"])

    qq_SRt1_p = np.array(predfile["TARGETS"][f"SR{method}t1"][f"{method}"])
    qq_SRt2_p = np.array(predfile["TARGETS"][f"SR{method}t2"][f"{method}"])

    q_ps = np.concatenate(
        (q_SRt1_p.reshape(-1, 1), q_SRt2_p.reshape(-1, 1)), axis=1
    )
    q_ps = ak.Array(q_ps)
    qq_ps = np.concatenate(
        (qq_SRt1_p.reshape(-1, 1), qq_SRt2_p.reshape(-1, 1)), axis=1
    )
    qq_ps = ak.Array(qq_ps)
    
    # jet detection probability
    dp_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["detection_probability"])
    dp_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["detection_probability"])
    # jet assignment probability
    ap_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["assignment_probability"])
    ap_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["assignment_probability"])

    dps = np.concatenate((dp_SRt1.reshape(-1, 1), dp_SRt2.reshape(-1, 1)), axis=1)
    aps = np.concatenate((ap_SRt1.reshape(-1, 1), ap_SRt2.reshape(-1, 1)), axis=1)
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

    fj_pt = np.array(testfile["INPUTS"]["BoostedJets"]["fj_pt"])
    fj_eta = np.array(testfile["INPUTS"]["BoostedJets"]["fj_eta"])
    fj_phi = np.array(testfile["INPUTS"]["BoostedJets"]["fj_phi"])
    fj_mass = np.array(testfile["INPUTS"]["BoostedJets"]["fj_mass"])
    fjs = ak.zip(
        {
            "pt": fj_pt,
            "eta": fj_eta,
            "phi": fj_phi,
            "mass": fj_mass,
        },
        with_name="Momentum4D",
    )


    # select predictions and targets
    q_ts_selected, qq_ts_selected, SRt_selected_pts = sel_target_SRt_by_mask(
        q_ts, qq_ts, SRt_pts, SRt_masks
    )
    q_ps_selected, qq_ps_selected = sel_pred_SRt_by_dp_ap(dps, aps, q_ps, qq_ps)


    # generate look up tables
    LUT_pred = gen_pred_SRt_LUT(
        q_ps_selected, qq_ps_selected,
        q_ts_selected, qq_ts_selected,
        js, fjs, 
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_SRt_LUT(
        q_ps_selected, qq_ps_selected,
        q_ts_selected, qq_ts_selected,
        SRt_selected_pts,
        ak.ArrayBuilder(),
    ).snapshot()


    return LUT_pred, LUT_target
