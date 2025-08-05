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

    # require b1 b2 assignment are AK4 jet
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

def sel_target_FRt_by_mask(b_ts, q1_ts, q2_ts, FRt_pts, FRt_overlap, FRt_masks):
    b_ts_selected = b_ts.mask[FRt_masks]
    b_ts_selected = ak.drop_none(b_ts_selected)

    q1_ts_selected = q1_ts.mask[FRt_masks]
    q1_ts_selected = ak.drop_none(q1_ts_selected)

    q2_ts_selected = q2_ts.mask[FRt_masks]
    q2_ts_selected = ak.drop_none(q2_ts_selected)

    FRt_selected_pts = FRt_pts.mask[FRt_masks]
    FRt_selected_pts = ak.drop_none(FRt_selected_pts)

    FRt_overlap_passed = FRt_overlap.mask[FRt_masks]
    FRt_overlap_passed = ak.drop_none(FRt_overlap_passed)

    return b_ts_selected, q1_ts_selected, q2_ts_selected, FRt_selected_pts, FRt_overlap_passed


# A pred look up table is in shape
# [event,
#    pred_FRt,
#       [correct_or_not, pt, overlap_w_FRt_reco, has_boost_FBt_target, which_FRt_target]]
@nb.njit
def gen_pred_FRt_LUT(
    b_ps_passed, q1_ps_passed, q2_ps_passed, 
    b_ts_selected, q1_ts_selected, q2_ts_selected, 
    js, goodJetIdx, FBt_overlap_selected, 
    builder
):
    # for each event
    for b_ps_e, q1_ps_e, q2_ps_e, b_ts_e, q1_ts_e, q2_ts_e, jets_e, goodJetIdx_e, FBt_overlap_e in zip(
        b_ps_passed, q1_ps_passed, q2_ps_passed, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        js, goodJetIdx, FBt_overlap_selected
    ):
        # for each predicted FRt assignment, check if any target t have a same FBt assignment
        builder.begin_list()

        for b_p, q1_p, q2_p in zip(b_ps_e, q1_ps_e, q2_ps_e):

            if (b_p in goodJetIdx_e) and (q1_p in goodJetIdx_e) and (q2_p in goodJetIdx_e):
                overlap = 0
            else:
                overlap = 1
            correct = 0
            has_t_FBt = -1
            FBt = -1

            predFRt_pt = (jets_e[b_p] + jets_e[q1_p] + jets_e[q2_p]).pt

            for i, (b_t, q1_t, q2_t, FBt_overlap) in enumerate(zip(b_ts_e, q1_ts_e, q2_ts_e, FBt_overlap_e)):
                if set((b_p, q1_p, q2_p)) == set((b_t, q1_t, q2_t)):
                    correct = 1
                    has_t_FBt = FBt_overlap
                    FBt = i

            builder.begin_list()
            builder.append(correct)
            builder.append(predFRt_pt)
            builder.append(overlap)
            builder.append(has_t_FBt)
            builder.append(FBt)
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
    FRt_pts, FBt_overlap_selected, 
    builder
):
    # for each event
    for b_ps_e, q1_ps_e, q2_ps_e, b_ts_e, q1_ts_e, q2_ts_e, FRt_pts_e, FBt_overlap_e in zip(
        b_ps_passed, q1_ps_passed, q2_ps_passed, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        FRt_pts, FBt_overlap_selected
    ):
        # for each target fatjet, check if the predictions have a p fatject same with the t fatjet
        builder.begin_list()

        for b_t, q1_t, q2_t, FRt_pt, FBt_overlap in zip(b_ts_e, q1_ts_e, q2_ts_e, FRt_pts_e, FBt_overlap_e):
            
            retrieved = 0
            can_boost_reco = FBt_overlap
            for b_p, q1_p, q2_p in zip(b_ps_e, q1_ps_e, q2_ps_e):
                if set((b_p, q1_p, q2_p)) == set((b_t, q1_t, q2_t)):
                    retrieved = 1

            builder.begin_list()
            builder.append(retrieved)
            builder.append(FRt_pt)
            builder.append(can_boost_reco)
            builder.end_list()

        builder.end_list()

    return builder


def parse_resolved_w_target(
    testfile, predfile, fjs_reco=None,
    overlap='Boosted',
    testfile_dict={'INPUTS': 'INPUTS', 'TARGETS': 'TARGETS'},
    predfile_dict={'INPUTS': 'INPUTS', 'TARGETS': 'TARGETS'},
):
    # FRt pt
    FRt1_pt = np.array(testfile[testfile_dict["TARGETS"]]["FRt1"]["pt"])
    FRt2_pt = np.array(testfile[testfile_dict["TARGETS"]]["FRt2"]["pt"])
    FRt_pts = np.concatenate((FRt1_pt.reshape(-1, 1), FRt2_pt.reshape(-1, 1)), axis=1)
    FRt_pts = ak.Array(FRt_pts)


    # resolved mask
    FRt1_mask = np.array(testfile[testfile_dict["TARGETS"]]["FRt1"]["mask"])
    FRt2_mask = np.array(testfile[testfile_dict["TARGETS"]]["FRt2"]["mask"])
    FRt_masks = np.concatenate((FRt1_mask.reshape(-1, 1), FRt2_mask.reshape(-1, 1)), axis=1)
    FRt_masks = ak.Array(FRt_masks)


    # boosted mask
    # SRqq
    SRqqt1_mask = np.array(testfile[testfile_dict["TARGETS"]]["SRqqt1"]["mask"])
    SRqqt2_mask = np.array(testfile[testfile_dict["TARGETS"]]["SRqqt2"]["mask"])
    SRqqt_masks = np.concatenate((SRqqt1_mask.reshape(-1, 1), SRqqt2_mask.reshape(-1, 1)), axis=1)
    SRqqt_masks = ak.Array(SRqqt_masks)
    # SRbq
    SRbqt1_mask = np.array(testfile[testfile_dict["TARGETS"]]["SRbqt1"]["mask"])
    SRbqt2_mask = np.array(testfile[testfile_dict["TARGETS"]]["SRbqt2"]["mask"])
    SRbqt_masks = np.concatenate((SRbqt1_mask.reshape(-1, 1), SRbqt2_mask.reshape(-1, 1)), axis=1)
    SRbqt_masks = ak.Array(SRbqt_masks)
    # FB
    FBt1_mask = np.array(testfile[testfile_dict["TARGETS"]]["FBt1"]["mask"])
    FBt2_mask = np.array(testfile[testfile_dict["TARGETS"]]["FBt2"]["mask"])
    FBt_masks = np.concatenate((FBt1_mask.reshape(-1, 1), FBt2_mask.reshape(-1, 1)), axis=1)
    FBt_masks = ak.Array(FBt_masks)

    # FR overlap with FB
    if overlap.lower() == 'boosted':
        FRt_overlap = FRt_masks & FBt_masks  # only FR / FB overlap
    elif overlap.lower() == 'all':
        FRt_overlap = FRt_masks & (SRqqt_masks | SRbqt_masks | FBt_masks)  # FR / all overlap
    else:
        raise Exception(f"Overlap criteria {overlap} not implemented, try \'Boosted\' or \'All\'.")
    FRt_overlap = ak.Array(ak.to_numpy(FRt_overlap).astype(float))  # necessary for downstream analysis, b/c NumPy requires uniform typing


    # target jets
    b_FRt1_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt1"]["b"])
    b_FRt2_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt2"]["b"])

    q1_FRt1_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt1"]["q1"])
    q1_FRt2_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt2"]["q1"])

    q2_FRt1_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt1"]["q2"])
    q2_FRt2_t = np.array(testfile[testfile_dict["TARGETS"]]["FRt2"]["q2"])

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
    b_FRt1_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["b"])
    b_FRt2_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["b"])

    q1_FRt1_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["q1"])
    q1_FRt2_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["q1"])

    q2_FRt1_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["q2"])
    q2_FRt2_p = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["q2"])

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
        dp_FRt1 = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["detection_probability"])
        dp_FRt2 = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["detection_probability"])
        # jet assignment probability
        ap_FRt1 = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["assignment_probability"])
        ap_FRt2 = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["assignment_probability"])
    except:
        # boosted Higgs detection probability
        dp_FRt1 = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["mask"]).astype("float")
        dp_FRt2 = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["mask"]).astype("float")

        # fatjet assignment probability
        ap_FRt1 = np.array(predfile[predfile_dict["TARGETS"]]["FRt1"]["mask"]).astype("float")
        ap_FRt2 = np.array(predfile[predfile_dict["TARGETS"]]["FRt2"]["mask"]).astype("float")

    dps = np.concatenate((dp_FRt1.reshape(-1, 1), dp_FRt2.reshape(-1, 1)), axis=1)
    aps = np.concatenate((ap_FRt1.reshape(-1, 1), ap_FRt2.reshape(-1, 1)), axis=1)
    # convert some numpy arrays to ak arrays
    dps = reset_collision_dp(dps, aps)


    # reconstruct jet 4-momentum objects
    j_pt = np.array(testfile[testfile_dict["INPUTS"]]["Jets"]["pt"])
    j_eta = np.array(testfile[testfile_dict["INPUTS"]]["Jets"]["eta"])
    j_phi = np.array(testfile[testfile_dict["INPUTS"]]["Jets"]["phi"])
    j_mass = np.array(testfile[testfile_dict["INPUTS"]]["Jets"]["mass"])
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
    b_ts_selected, q1_ts_selected, q2_ts_selected, FRt_selected_pts, overlap_selected = sel_target_FRt_by_mask(
        b_ts, q1_ts, q2_ts, FRt_pts, FRt_overlap, FRt_masks
    )
    b_ps_selected, q1_ps_selected, q2_ps_selected = sel_pred_FRt_by_dp_ap(dps, aps, b_ps, q1_ps, q2_ps)


    # find jets that are overlapped with reco boosted top
    if fjs_reco is None:
        goodJetIdx = ak.local_index(js)
    else:
        goodJetIdx = get_unoverlapped_jet_index(fjs_reco, js, dR_min=0.8)


    # generate look up tables
    LUT_pred = gen_pred_FRt_LUT(
        b_ps_selected, q1_ps_selected, q2_ps_selected, 
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        js, goodJetIdx, overlap_selected, 
        ak.ArrayBuilder()
    ).snapshot()
    LUT_target = gen_target_FRt_LUT(
        b_ps_selected, q1_ps_selected, q2_ps_selected,
        b_ts_selected, q1_ts_selected, q2_ts_selected, 
        FRt_selected_pts, overlap_selected,
        ak.ArrayBuilder(),
    ).snapshot()


    return LUT_pred, LUT_target, goodJetIdx
