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

def get_unoverlapped_jet_index(fjs, js, dR_min=0.5):
    overlapped = ak.sum(js[:, np.newaxis].deltaR(fjs) < dR_min, axis=-2) > 0
    jet_index_passed = ak.local_index(js).mask[~overlapped]
    jet_index_passed = ak.drop_none(jet_index_passed)
    return jet_index_passed


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

def sel_target_SRt_by_mask(q_ts, qq_ts, SRt_pts, SRt_overlap, SRt_masks):
    q_ts_selected = q_ts.mask[SRt_masks]
    q_ts_selected = ak.drop_none(q_ts_selected)

    qq_ts_selected = qq_ts.mask[SRt_masks]
    qq_ts_selected = ak.drop_none(qq_ts_selected)

    SRt_selected_pts = SRt_pts.mask[SRt_masks]
    SRt_selected_pts = ak.drop_none(SRt_selected_pts)

    SRt_overlap_passed = SRt_overlap.mask[SRt_masks]
    SRt_overlap_passed = ak.drop_none(SRt_overlap_passed)

    return q_ts_selected, qq_ts_selected, SRt_selected_pts, SRt_overlap_passed


# A pred look up table is in shape
# [event,
#    pred_SRt,
#       [correct_or_not, pt, overlap_w_SRt_reco, has_boost_FBt_target, which_SRt_target]]
@nb.njit
def gen_pred_SRt_LUT(
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

            if (q_p in goodJetIdx_e) and (qq_p in goodFatJetIdx_e):
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
def gen_target_SRt_LUT(
    q_ps_passed, qq_ps_passed,
    q_ts_selected, qq_ts_selected,
    SRt_pts, FBt_overlap_selected, 
    builder
):
    # for each event
    for q_ps_e, qq_ps_e, q_ts_e, qq_ts_e, SRt_pts_e, FBt_overlap_e in zip(
        q_ps_passed, qq_ps_passed,
        q_ts_selected, qq_ts_selected,
        SRt_pts, FBt_overlap_selected
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        for q_t, qq_t, SRt_pt, FBt_overlap in zip(q_ts_e, qq_ts_e, SRt_pts_e, FBt_overlap_e):
            
            retrieved = 0
            can_boost_reco = FBt_overlap
            for q_p, qq_p in zip(q_ps_e, qq_ps_e):
                if set((q_p, qq_p - N_AK4_JETS)) == set((q_t, qq_t)):
                    retrieved = 1

            builder.begin_list()
            builder.append(retrieved)
            builder.append(SRt_pt)
            builder.append(can_boost_reco)
            builder.end_list()

        builder.end_list()

    return builder


def parse_semi_resolved_w_target(
    testfile, predfile, method,
    vfjs_reco=None
):
    # FRt pt
    SRt1_pt = np.array(testfile["TARGETS"][f"SR{method}t1"]["pt"])
    SRt2_pt = np.array(testfile["TARGETS"][f"SR{method}t1"]["pt"])
    SRt_pts = np.concatenate((SRt1_pt.reshape(-1, 1), SRt2_pt.reshape(-1, 1)), axis=1)
    SRt_pts = ak.Array(SRt_pts)


    # resolved mask
    SRt1_mask = np.array(testfile["TARGETS"][f"SR{method}t1"]["mask"])
    SRt2_mask = np.array(testfile["TARGETS"][f"SR{method}t1"]["mask"])
    SRt_masks = np.concatenate((SRt1_mask.reshape(-1, 1), SRt2_mask.reshape(-1, 1)), axis=1)
    SRt_masks = ak.Array(SRt_masks)


    # boosted mask
    # FB
    FBt1_mask = np.array(testfile["TARGETS"]["FBt1"]["mask"])
    FBt2_mask = np.array(testfile["TARGETS"]["FBt2"]["mask"])
    FBt_masks = np.concatenate((FBt1_mask.reshape(-1, 1), FBt2_mask.reshape(-1, 1)), axis=1)
    FBt_masks = ak.Array(FBt_masks)

    # FR overlap with FB
    if method == 'bq':
        # SRqq
        SRqqt1_mask = np.array(testfile["TARGETS"]["SRqqt1"]["mask"])
        SRqqt2_mask = np.array(testfile["TARGETS"]["SRqqt2"]["mask"])
        SRqqt_masks = np.concatenate((SRqqt1_mask.reshape(-1, 1), SRqqt2_mask.reshape(-1, 1)), axis=1)
        SRqqt_masks = ak.Array(SRqqt_masks)
        
        SRt_overlap = SRt_masks & (SRqqt_masks | FBt_masks)
    elif method == 'qq':
        SRt_overlap = SRt_masks & FBt_masks
    else:
        raise Exception(f"Semi-resolved method \'{method}\' not implemented, try \'bq\' or \'qq\'.")
    SRt_overlap = ak.Array(ak.to_numpy(SRt_overlap).astype(float))  # necessary for downstream analysis, b/c NumPy requires uniform typing


    # target jets
    q_SRt1_t = np.array(testfile["TARGETS"][f"SR{method}t1"][f"{method[0]}"])
    q_SRt2_t = np.array(testfile["TARGETS"][f"SR{method}t2"][f"{method[0]}"])

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
    q_SRt1_p = np.array(predfile["TARGETS"][f"SR{method}t1"][f"{method[0]}"])
    q_SRt2_p = np.array(predfile["TARGETS"][f"SR{method}t2"][f"{method[0]}"])

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

    try:
        # jet detection probability
        dp_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["detection_probability"])
        dp_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["detection_probability"])
        # jet assignment probability
        ap_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["assignment_probability"])
        ap_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["assignment_probability"])
    except:
        # semi-boosted top detection probability
        dp_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["mask"]).astype("float")
        dp_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["mask"]).astype("float")

        # jet/fatjet assignment probability
        ap_SRt1 = np.array(predfile["TARGETS"][f"SR{method}t1"]["mask"]).astype("float")
        ap_SRt2 = np.array(predfile["TARGETS"][f"SR{method}t2"]["mask"]).astype("float")

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

    fj_pt = np.array(testfile["INPUTS"]["Jets"]["pt"])
    fj_eta = np.array(testfile["INPUTS"]["Jets"]["eta"])
    fj_phi = np.array(testfile["INPUTS"]["Jets"]["phi"])
    fj_mass = np.array(testfile["INPUTS"]["Jets"]["mass"])
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
    q_ts_selected, qq_ts_selected, SRt_selected_pts, overlap_selected = sel_target_SRt_by_mask(
        q_ts, qq_ts, SRt_pts, SRt_overlap, SRt_masks
    )
    q_ps_selected, qq_ps_selected = sel_pred_SRt_by_dp_ap(dps, aps, q_ps, qq_ps)


    # find jets that are overlapped with reco boosted top
    if vfjs_reco is None:
        goodJetIdx = ak.local_index(js)
        goodFatJetIdx = ak.local_index(fjs)
    else:
        goodJetIdx = get_unoverlapped_jet_index(vfjs_reco, js, dR_min=0.8)
        goodFatJetIdx = get_unoverlapped_jet_index(vfjs_reco, fjs, dR_min=0.8)


    # generate look up tables
    LUT_pred = gen_pred_SRt_LUT(
        q_ps_selected, qq_ps_selected,
        q_ts_selected, qq_ts_selected,
        js, goodJetIdx, 
        fjs, goodFatJetIdx,
        overlap_selected, 
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_SRt_LUT(
        q_ps_selected, qq_ps_selected,
        q_ts_selected, qq_ts_selected,
        SRt_selected_pts, 
        overlap_selected,
        ak.ArrayBuilder(),
    ).snapshot()

    # reconstruct FBt to remove overlapped ak4 & ak8 jets
    fj_reco = fjs[qq_ps_selected - N_AK4_JETS]


    return LUT_pred, LUT_target, fj_reco
