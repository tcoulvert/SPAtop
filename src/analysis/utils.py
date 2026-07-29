import itertools

import awkward as ak
import numba as nb
import numpy as np
from hist.intervals import clopper_pearson_interval


def n_alpha(string: str):
    return len([c for c in string if c.isalpha()])


def reset_collision_dp(dps, aps):
    ap_filter = aps < 1 / (13 * 13)
    return ak.where(ap_filter, 0, dps)


def get_symmetries(recos, jet_labels):
    symmetries = []
    for reco in recos:
        unique_matches = [(i, i) for i, label in enumerate(jet_labels[reco]) if label.isalpha()]
        sym_matches = [i for i, label in enumerate(jet_labels[reco]) if not label.isalpha()]
        reco_syms = np.array([
            np.vstack([np.array(unique_matches, dtype=int), np.hstack([np.array(sym_matches, dtype=int)[:, np.newaxis], np.array(sym_combo, dtype=int)[:, np.newaxis]])])
            for sym_combo in list(itertools.permutations(sym_matches, r=len(sym_matches)))
        ], dtype=int)
        symmetries.append(reco_syms)
    return symmetries


@nb.njit
def match_jet(jet1, jet2):
    if jet1 is None or jet2 is None: return False
    else: return (
        jet1.pt == jet2.pt and jet1.eta == jet2.eta 
        and jet1.phi == jet2.phi and jet1.mass == jet2.mass
    )

@nb.njit
def deltaR(jet1, jet2):
    if jet1 is None or jet2 is None: return 999

    deltaPhi = jet1.phi - jet2.phi
    if deltaPhi < -np.pi: deltaPhi = deltaPhi + 2*np.pi
    elif deltaPhi > np.pi: deltaPhi = deltaPhi - 2*np.pi

    deltaEta = jet1.eta - jet2.eta
    if deltaEta < 0: deltaEta = -1 * deltaEta

    deltaR = (deltaPhi**2 + deltaEta**2)**0.5
    return deltaR

@nb.njit
def overlap(jets, idxs, nrecos, ntops, deltaRs, builder):
    for jets_event, idx_event, nrecos_event in zip(jets, idxs, nrecos):
        builder.begin_list()

        good_idxs = []
        for idx in idx_event[:nrecos_event]:
            if len(good_idxs) == ntops: break
            elif jets_event[idx] is None: continue
            else:
                append = True
                for k, jet in enumerate(jets_event[idx]):
                    if jet is None: append = False; break
                    for idx_ in good_idxs:
                        for k_, jet_ in enumerate(jets_event[idx_]):
                            delR = deltaR(jet, jet_)
                            if delR < deltaRs[idx][k] or delR < deltaRs[idx_][k_]: append = False; break
                if append: good_idxs.append(idx)

        for good_idx in good_idxs: builder.append(good_idx)
        builder.end_list()

    return builder

def reco_reorder(predicted_jets, dps, aps, n_recos, n_tops, deltaRs):
    ps = dps * aps
    idx_sort = np.flip(np.argsort(ps, axis=-1), axis=-1)
    idx_sel = overlap(predicted_jets, idx_sort, n_recos, n_tops, ak.Array(deltaRs), ak.ArrayBuilder()).snapshot()

    return idx_sel


def dp_to_TopNumProb(dps):
    # get maximum number of targets
    Noptions = ak.max(ak.num(dps, axis=-1), axis=None)

    # prepare a list for constructing [P_0t, P_1t, P_2t, ...]
    probs = []

    # loop through all possible number of existing targets
    for N in range(Noptions + 1):
        # get all combinations of targets
        combs = list(itertools.combinations(range(Noptions), N))

        # calculate the probability of N particles existing for each combination
        P_exist_per_comb = [np.prod(dps[:, list(comb)], axis=-1) for comb in combs]

        # calculate the probability of Nmax-N particles not existing for each  combination
        P_noexist_per_comb = [np.prod(1 - dps[:, list(set(range(Noptions)) - set(comb))], axis=-1) for comb in combs]

        # concatenate each combination to array for further calculation
        P_exist_per_comb = np.concatenate([
            np.reshape(P_comb_e, newshape=(-1, 1)) 
            for P_comb_e in P_exist_per_comb
        ], axis=1)
        P_noexist_per_comb = np.concatenate([
            np.reshape(P_comb_e, newshape=(-1, 1)) 
            for P_comb_e in P_noexist_per_comb
        ], axis=1)

        # for each combination, calculate the joint probability
        #  of N particles existing and Nmax-N not existing
        P_per_comb = P_exist_per_comb * P_noexist_per_comb

        # sum over all possible configurations of N existing and Nmax-N not existing
        P = np.sum(P_per_comb, axis=-1)

        # reshape and add to the prob list
        probs.append(np.reshape(P, newshape=(-1, 1)))

    # convert the probs list to arr
    probs_arr = np.concatenate(probs, axis=1)

    return probs_arr


# calculate purity/efficiency
def calc_pureff(LUT, bins):
    if LUT is None: return None, None

    Tops = np.array([top for top in LUT])

    Tops_inds = np.digitize(Tops[:, 1], bins)  # index 1 is pt

    correctTruth_per_bin = []
    for bin_i in range(1, len(bins) + 1):
        correctTruth_per_bin.append(Tops[:, 0][Tops_inds == bin_i])  # index 0 is correct prediction
    correctTruth_per_bin = ak.Array(correctTruth_per_bin)

    means = ak.mean(correctTruth_per_bin, axis=-1)

    errs = np.abs(
        clopper_pearson_interval(num=ak.sum(correctTruth_per_bin, axis=-1), denom=ak.num(correctTruth_per_bin, axis=-1))
        - means
    )

    return means, errs
