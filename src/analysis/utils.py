import itertools

import awkward as ak
import numpy as np
from hist.intervals import clopper_pearson_interval


def reset_collision_dp(dps, aps):
    ap_filter = aps < 1 / (13 * 13)
    dps_reset = dps
    dps_reset[ap_filter] = 0
    return dps_reset


def dp_to_TopNumProb(dps):
    # get maximum number of targets
    Nmax = dps.shape[-1]

    # prepare a list for constructing [P_0t, P_1t, P_2t, ...]
    probs = []

    # loop through all possible number of existing targets
    for N in range(Nmax + 1):
        # get all combinations of targets
        combs = list(itertools.combinations(range(Nmax), N))

        # calculate the probability of N particles existing for each combination
        P_exist_per_comb = [np.prod(dps[:, list(comb)], axis=-1) for comb in combs]

        # calculate the probability fo Nmax-N particles not existing for each  combination
        P_noexist_per_comb = [np.prod(1 - dps[:, list(set(range(Nmax)) - set(comb))], axis=-1) for comb in combs]

        # concatenate each combination to array for further calculation
        P_exist_per_comb = [np.reshape(P_comb_e, newshape=(-1, 1)) for P_comb_e in P_exist_per_comb]
        P_exist_per_comb = np.concatenate(P_exist_per_comb, axis=1)
        P_noexist_per_comb = [np.reshape(P_comb_e, newshape=(-1, 1)) for P_comb_e in P_noexist_per_comb]
        P_noexist_per_comb = np.concatenate(P_noexist_per_comb, axis=1)

        # for each combination, calculate the joint probability
        # of N particles existing and Nmax-N not existing
        P_per_comb = P_exist_per_comb * P_noexist_per_comb

        # sum over all possible configurations of N existing and Nmax-N not existing
        P = np.sum(P_per_comb, axis=-1)

        # reshape and add to the prob list
        probs.append(np.reshape(P, newshape=(-1, 1)))

    # convert the probs list to arr
    probs_arr = np.concatenate(probs, axis=1)

    return probs_arr


# calculate efficiency
# if bins=None, put all data in a single bin
def calc_eff(LUT_boosted_pred, LUT_resolved_pred, bins):

    predTops = []

    if LUT_boosted_pred is not None:
        # boosted H don't need post processing
        predTops_boosted = [predTop for event in LUT_boosted_pred for predTop in event]
        predTops += predTops_boosted

    if LUT_resolved_pred is not None:
        if LUT_boosted_pred is not None:
            # calculate merged efficiency
            # Remove overlapped resolved Top_reco
            predTops_resolved = [predTop[0:2] for event in LUT_resolved_pred for predTop in event if predTop[2] == 0]
            predTops += predTops_resolved
        else:
            # calculate resolved efficiency
            predTops_resolved = [predTop[0:2] for event in LUT_resolved_pred for predTop in event]
            predTops += predTops_resolved

    # then merge into the list with their pT
    predTops = np.array(predTops)

    predTops_inds = np.digitize(predTops[:, 1], bins)

    correctTruth_per_bin = []
    for bin_i in range(1, len(bins) + 1):
        correctTruth_per_bin.append(predTops[:, 0][predTops_inds == bin_i])
    correctTruth_per_bin = ak.Array(correctTruth_per_bin)

    means = ak.mean(correctTruth_per_bin, axis=-1)

    errs = np.abs(
        clopper_pearson_interval(num=ak.sum(correctTruth_per_bin, axis=-1), denom=ak.num(correctTruth_per_bin, axis=-1))
        - means
    )

    return means, errs


# calculate purity
def calc_pur(LUT_boosted_target, LUT_resolved_target, bins):

    targetTops = []

    if LUT_boosted_target is not None:
        # boosted Top don't need post processing
        targetTops_boosted = [targetTop for event in LUT_boosted_target for targetTop in event]
        targetTops += targetTops_boosted

    if LUT_resolved_target is not None:
        if LUT_boosted_target is not None:
            # calculate merged purity
            # only consider resolved target Top that doesn't have a corresponding boosted Top target
            targetTops_resolved = [targetTop[0:2] for event in LUT_resolved_target for targetTop in event if targetTop[2] == 0]
            targetTops += targetTops_resolved
        else:
            # calculate resolved only purity
            targetTops_resolved = [targetTop[0:2] for event in LUT_resolved_target for targetTop in event]
            targetTops += targetTops_resolved

    targetTops = np.array(targetTops)

    targetTops_inds = np.digitize(targetTops[:, 1], bins)

    correctTruth_per_bin = []
    for bin_i in range(1, len(bins) + 1):
        correctTruth_per_bin.append(targetTops[:, 0][targetTops_inds == bin_i])
    correctTruth_per_bin = ak.Array(correctTruth_per_bin)

    means = ak.mean(correctTruth_per_bin, axis=-1)

    errs = np.abs(
        clopper_pearson_interval(num=ak.sum(correctTruth_per_bin, axis=-1), denom=ak.num(correctTruth_per_bin, axis=-1))
        - means
    )

    return means, errs
