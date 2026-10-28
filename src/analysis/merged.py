import itertools

import awkward as ak
import numba as nb
import numpy as np
import vector
vector.register_awkward()

from src.analysis.utils import reco_reorder, reset_collision_dp, dp_to_TopNumProb, match_jet, get_symmetries, n_alpha, get_numerical, get_jets, get_jet4moms

N_AK5_JETS = 10
N_AK8_JETS = 3
N_TOPS = 2
DELTARS = None
SYMMETRIES = None
LUT_FILL_VALUE = -999

def sel_target_t_by_mask(target_jets, target_pts, target_masks):
    filter = target_masks
    selected_target_jets = ak.mask(target_jets, filter)
    selected_target_pts = ak.where(filter, target_pts, LUT_FILL_VALUE)

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
    selected_predicted_pts = ak.where(filter, predicted_pts, LUT_FILL_VALUE)
    selected_predicted_dps = ak.where(filter, dps, LUT_FILL_VALUE)
    selected_predicted_aps = ak.where(filter, dps, LUT_FILL_VALUE)

    return selected_predicted_jets, selected_predicted_pts, selected_predicted_dps, selected_predicted_aps, idx_sel


# A look up table is in shape
# [event x N_tops,
#        [valid_top, exact_match_pred, class_match_pred, gen_pt, reco_pt, dp, ap]]
def generate_LUT(predicted_jets, target_jets, predicted_toppt, target_toppt, predicted_dp, predicted_ap, selected_order):
    return ak.to_numpy(nbgenerate_LUT(
        predicted_jets, target_jets, predicted_toppt, target_toppt, predicted_dp, predicted_ap,
        selected_order, SYMMETRIES,
        ak.ArrayBuilder()
    ).snapshot(), allow_missing=False)

@nb.njit
def values(nbdict: nb.typed.Dict):
    return nb.typed.List([val for _, val in nbdict.items()])

@nb.njit
def nbgenerate_LUT(
    predicted_jets, target_jets, predicted_toppt, target_toppt, predicted_dp, predicted_ap,
    selected_order, symmetries,
    builder
):
    # for each event
    for pjets_event, tjets_event, predpt_event, targpt_event, preddp_event, predap_event, order_event in zip(
        predicted_jets, target_jets,
        predicted_toppt, target_toppt,
        predicted_dp, predicted_ap,
        selected_order
    ):
        # For each target top build separate valid and invalid maps [top index -> top pt (-999.0 when no valid top)]
        valid_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)
        invalid_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.float64)
        for top_idx in range(N_TOPS):
            # Check for any valid targets
            for target_idx in range(top_idx, len(tjets_event), N_TOPS):
                tjets, tpt = tjets_event[target_idx], targpt_event[target_idx]
                if tjets is None: continue
                valid_tops[top_idx] = tpt; break
            if top_idx not in valid_tops: 
                invalid_tops[top_idx] = LUT_FILL_VALUE

        # For each valid top attempt to match to the predicted tops, prioritizing exact matches, followed by class matches, followed by any valid top
        exact_matched_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
        for valid_top_idx in valid_tops.keys():
            # Exact matches
            for pred_idx in order_event:
                pjets = pjets_event[pred_idx]
                if pred_idx in values(exact_matched_tops): continue #prevents candidate that exactly matches both tops' targets to be credited to both
                target_idx = ([pred_idx - i for i in range(1, (pred_idx % N_TOPS)+1)][::-1]+[pred_idx]+[pred_idx + i for i in range(1, N_TOPS-(pred_idx % N_TOPS))])[valid_top_idx]
                tjets = tjets_event[target_idx]
                if pjets is None or tjets is None: continue
                for symand in symmetries[pred_idx]:
                    n_matched_jets = 0
                    for labels in symand:
                        if match_jet(pjets[labels[0]], tjets[labels[1]]): n_matched_jets += 1
                    if n_matched_jets == len(symand):
                        exact_matched_tops[valid_top_idx] = pred_idx
                        break
                if valid_top_idx in exact_matched_tops: break #prevents candidate that exactly matches both tops' targets to be credited to both
                

        class_matched_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
        for valid_top_idx in valid_tops.keys():
            if valid_top_idx in exact_matched_tops: continue
            # Class matches
            for pred_idx in order_event:
                pjets = pjets_event[pred_idx]
                ###if pred_idx in values(exact_matched_tops): continue 
                #needs to also skip candidates pass already used
                if pred_idx in values(exact_matched_tops) or pred_idx in values(class_matched_tops): continue
                target_idx = ([pred_idx - i for i in range(1, (pred_idx % N_TOPS)+1)][::-1]+[pred_idx]+[pred_idx + i for i in range(1, N_TOPS-(pred_idx % N_TOPS))])[valid_top_idx]
                tjets = tjets_event[target_idx]
                if pjets is None or tjets is None: continue
                class_matched_tops[valid_top_idx] = pred_idx
                break #stop at first acceptable canidate, otherwise loop always ends on worst rank top

        valid_matched_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
        for valid_top_idx in valid_tops.keys():
            if valid_top_idx in exact_matched_tops or valid_top_idx in class_matched_tops: continue
            # Valid matches
            for pred_idx in order_event:
                pjets = pjets_event[pred_idx]
                ###if pred_idx in values(exact_matched_tops) or pred_idx in values(class_matched_tops): continue
                #same thing: needs to skip candidates already used
                if pred_idx in values(exact_matched_tops) or pred_idx in values(class_matched_tops) or pred_idx in values(valid_matched_tops): continue
                if pjets is None: continue
                valid_matched_tops[valid_top_idx] = pred_idx
                break #stop at first acceptable canidate, otherwise loop always ends on worst rank top

        # For each invalid top match any remaining predicted tops
        invalid_matched_tops = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64)
        for invalid_top_idx in invalid_tops.keys():
            for pred_idx in order_event:
                pjets = pjets_event[pred_idx]
                ###if pred_idx in values(exact_matched_tops) or pred_idx in values(class_matched_tops) or pred_idx in values(valid_matched_tops): continue
                if pred_idx in values(exact_matched_tops) or pred_idx in values(class_matched_tops) or pred_idx in values(valid_matched_tops) or pred_idx in values(invalid_matched_tops): continue
                if pjets is None: continue
                invalid_matched_tops[invalid_top_idx] = pred_idx
                break

        # Place results into Ak Builder
        builder.begin_list()
        for top_idx in range(N_TOPS):
            builder.begin_list()
            if top_idx in exact_matched_tops:
                builder.append(1)                                            # Valid top is true
                builder.append(1)                                            # Valid top was matched to prediction
                builder.append(1)                                            # Valid top was matched to prediction's reco class
                builder.append(1)                                            # Top was predicted
                builder.append(valid_tops[top_idx])                          # Gen pt
                builder.append(predpt_event[exact_matched_tops[top_idx]])    # Reco pt
                builder.append(preddp_event[exact_matched_tops[top_idx]])    # Reco detection probability
                builder.append(predap_event[exact_matched_tops[top_idx]])    # Reco assignment probability
            elif top_idx in class_matched_tops:
                builder.append(1)                                            # Valid top is true
                builder.append(0)                                            # Valid top was NOT matched to prediction
                builder.append(1)                                            # Valid top was matched to prediction's reco class
                builder.append(1)                                            # Top was predicted
                builder.append(valid_tops[top_idx])                          # Gen pt
                builder.append(predpt_event[class_matched_tops[top_idx]])    # Reco pt
                builder.append(preddp_event[class_matched_tops[top_idx]])    # Reco detection probability
                builder.append(predap_event[class_matched_tops[top_idx]])    # Reco assignment probability
            elif top_idx in valid_matched_tops:
                builder.append(1)                                            # Valid top is true
                builder.append(0)                                            # Valid top was NOT matched to prediction
                builder.append(0)                                            # Valid top was NOT matched to prediction's reco class
                builder.append(1)                                            # Top was predicted
                builder.append(valid_tops[top_idx])                          # Gen pt
                builder.append(predpt_event[valid_matched_tops[top_idx]])    # Reco pt
                builder.append(preddp_event[valid_matched_tops[top_idx]])    # Reco detection probability
                builder.append(predap_event[valid_matched_tops[top_idx]])    # Reco assignment probability
            elif top_idx in invalid_matched_tops:
                builder.append(0)                                            # Valid top is NOT true
                builder.append(0)                                            # Valid top was NOT matched to prediction
                builder.append(0)                                            # Valid top was NOT matched to prediction's reco class
                builder.append(1)                                            # Top was predicted
                builder.append(invalid_tops[top_idx])                        # Gen pt
                builder.append(predpt_event[invalid_matched_tops[top_idx]])  # Reco pt
                builder.append(preddp_event[invalid_matched_tops[top_idx]])  # Reco detection probability
                builder.append(predap_event[invalid_matched_tops[top_idx]])  # Reco assignment probability
            elif top_idx in valid_tops:
                builder.append(1)                                            # Valid top is true
                builder.append(0)                                            # Valid top was NOT matched to prediction
                builder.append(0)                                            # Valid top was NOT matched to prediction's reco class
                builder.append(0)                                            # Top was NOT predicted
                builder.append(valid_tops[top_idx])                          # Gen pt
                builder.append(LUT_FILL_VALUE)                               # Reco pt
                builder.append(LUT_FILL_VALUE)                               # Reco detection probability
                builder.append(LUT_FILL_VALUE)                               # Reco assignment probability
            elif top_idx in invalid_tops:
                builder.append(0)                                            # Valid top is NOT true
                builder.append(0)                                            # Valid top was NOT matched to prediction
                builder.append(0)                                            # Valid top was NOT matched to prediction's reco class
                builder.append(0)                                            # Top was NOT predicted
                builder.append(invalid_tops[top_idx])                        # Gen pt
                builder.append(LUT_FILL_VALUE)                               # Reco pt
                builder.append(LUT_FILL_VALUE)                               # Reco detection probability
                builder.append(LUT_FILL_VALUE)                               # Reco assignment probability
            else:
                raise Exception(f"Unknown category of gen-reco matching")
            builder.end_list()
        builder.end_list()
    return builder


def parse_merged_w_target(
    testfile, predfile, reco_regex: str='', chi2: bool=False
):  
    # handle different pl version
    if "TARGETS" not in predfile.keys():
        predfile["INPUTS"] = predfile["SpecialKey.Inputs"]
        predfile["TARGETS"] = predfile["SpecialKey.Targets"]
    
    print(f"Processing reco: {reco_regex}")
    global N_TOPS, N_AK5_JETS, N_AK8_JETS, DELTARS, SYMMETRIES
    reconstructions = sorted([key for key in predfile["TARGETS"].keys() if reco_regex in key])
    if len(reconstructions) == 0: return None
    N_TOPS = np.max([int(k) for key in reconstructions for k in key if k.isdigit()])
    print(f"Number of tops: {N_TOPS}")
    jet_labels = {reco: sorted([key for key in predfile["TARGETS"][reco].keys() if 'prob' not in key]) for reco in reconstructions}
    DELTARS = [[0.8 if n_alpha(label) > 1 else 0.5 for label in jet_labels[reco]] for reco in reconstructions]
    SYMMETRIES = get_symmetries(reconstructions, jet_labels)

    # jet 4-momentums
    jets, fatjets = get_jet4moms(testfile)
    N_AK5_JETS = ak.max(ak.local_index(jets), axis=None) + 1
    print(f"Number of AK5 jets: {N_AK5_JETS}")
    N_AK8_JETS = ak.max(ak.local_index(fatjets), axis=None) + 1
    print(f"Number of AK8 jets: {N_AK8_JETS}")

    # target pt
    target_pts = get_numerical(testfile, "pt", reconstructions)

    # target MASK
    target_masks = get_numerical(testfile, "MASK", reconstructions)

    # target jets
    target_jets = get_jets(testfile, reconstructions, jet_labels, jets, fatjets)

    # predicted jets
    predicted_jets = get_jets(predfile, reconstructions, jet_labels, jets, fatjets)
    predicted_pts = ak.Array(ak.sum(predicted_jets, axis=-1).pt)

    # predicted probabilities
    dps = get_numerical(predfile, "detection_probability", reconstructions)
    aps = get_numerical(predfile, "assignment_probability", reconstructions)
    if not chi2: dps = reset_collision_dp(dps, aps)

    # select predictions and targets
    selected_target_jets, selected_target_pts = sel_target_t_by_mask(target_jets, target_pts, target_masks)
    selected_predicted_jets, selected_predicted_pts, selected_predicted_dps, selected_predicted_aps, selected_order = sel_pred_t_by_prob(predicted_jets, predicted_pts, dps, aps)
    
    # generate look up tables
    LUT = generate_LUT(
        selected_predicted_jets, selected_target_jets, 
        selected_predicted_pts, selected_target_pts, 
        selected_predicted_dps, selected_predicted_aps, 
        selected_order
    )

    return LUT
