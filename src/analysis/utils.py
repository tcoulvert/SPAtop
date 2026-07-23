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
def calc_eff(LUT_pred, bins):

    predTops = np.array([predTop for event in LUT_pred for predTop in event])

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
def calc_pur(LUT_target, bins):

    targetTops = np.array([targetTop for event in LUT_target for targetTop in event])

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
