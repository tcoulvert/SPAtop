"""Unit tests for the gen-to-reco matching functions in matching.py: the
scoring functions and their gates (angular, mass-window, pT), the overlap
predicates, the jet-to-fat-jet matcher, and reconstruct_top's behaviour when
nothing qualifies.

The scoring functions are njit kernels that production only ever calls from
inside reconstruct_top, where every gen particle has been pulled out of an
awkward array and so arrives as a vector object. Calling them from Python
with bare ak.Record arguments types differently and fails, so the tests go
through the small njit shim `_apply` that indexes the arrays the same way
reconstruct_top does."""
import functools
import operator

import awkward as ak
import numba as nb
import numpy as np
import pytest
import vector

from src.data.delphes import matching as M

# A top whose daughters lie inside the FR mass windows (checked by the fixture):
# b at (eta 0.7, phi 0.9), q1 at (-0.7, 0.9), q2 at (0.0, 1.5).
B, Q1, Q2 = (110.0, 0.7, 0.9, 4.8), (70.0, -0.7, 0.9, 0.3), (70.0, 0.0, 1.5, 0.3)


def vobj(kin):
    return vector.obj(pt=kin[0], eta=kin[1], phi=kin[2], mass=kin[3])


def vsum(*kins):
    """(pt, eta, phi, mass) of the four-vector sum: gen W and top records carry their own kinematics."""
    v = functools.reduce(operator.add, map(vobj, kins))
    return (v.pt, v.eta, v.phi, v.mass)


@nb.njit
def _apply(func, tops, bs, ws, q1s, q2s, cands):
    return func(tops[0][0], bs[0][0], ws[0][0], q1s[0][0], q2s[0][0], cands[0][0])


def score(func, momenta, gen, cand):
    """gen = (top, b, W, q1, q2) kinematic tuples; cand = a [1, 1] jagged array of candidate records."""
    return _apply(func, *(momenta([k]) for k in gen), cand)


def fr_cand(momenta, b, q1, q2):
    return ak.zip({"bjet": momenta([b]), "q1jet": momenta([q1]), "q2jet": momenta([q2])})


def sr_cand(momenta, jet, fjet):
    return ak.zip({"jet": momenta([jet]), "fjet": momenta([fjet])})


def fb_cand(momenta, fjet):
    return ak.zip({"fjet": momenta([fjet])})


@pytest.fixture
def fr_gen():
    top, w = vsum(B, Q1, Q2), vsum(Q1, Q2)
    assert abs(top[3] - M.TOP_MASS) < M.TOP_MASS_WINDOW and abs(w[3] - M.W_MASS) < M.W_MASS_WINDOW
    return (top, B, w, Q1, Q2)


# ------------------------------------------------------------ fully resolved
def test_fully_resolved_perfect_match_scores_zero(momenta, fr_gen):
    assert score(M.FullyResolved_top, momenta, fr_gen, fr_cand(momenta, B, Q1, Q2)) == pytest.approx(0.0, abs=1e-6)


def test_fully_resolved_score_is_the_summed_delta_r(momenta, fr_gen):
    b_off = (B[0], B[1], B[2] + 0.3, B[3])
    q1_off = (Q1[0], Q1[1] + 0.2, Q1[2], Q1[3])
    assert score(M.FullyResolved_top, momenta, fr_gen, fr_cand(momenta, b_off, q1_off, Q2)) == pytest.approx(0.5, abs=1e-6)


def test_fully_resolved_rejects_a_jet_outside_the_cone(momenta, fr_gen):
    q2_far = (Q2[0], Q2[1], Q2[2] + 0.6, Q2[3])                      # 0.6 > JET_DR
    assert score(M.FullyResolved_top, momenta, fr_gen, fr_cand(momenta, B, Q1, q2_far)) == M.DR_FILL_VALUE


def test_fully_resolved_rejects_top_mass_outside_window(momenta, fr_gen):
    heavy_b = (1500.0,) + B[1:]                                      # same direction, m(bqq) >> 242.5
    assert score(M.FullyResolved_top, momenta, fr_gen, fr_cand(momenta, heavy_b, Q1, Q2)) == M.DR_FILL_VALUE


def test_fully_resolved_rejects_w_mass_outside_window(momenta, fr_gen):
    heavy_q2 = (700.0,) + Q2[1:]                                     # same direction, m(qq) >> 110
    assert score(M.FullyResolved_top, momenta, fr_gen, fr_cand(momenta, B, Q1, heavy_q2)) == M.DR_FILL_VALUE


# ------------------------------------------------------------ semi resolved
def test_semi_resolved_qq_gates(momenta):
    bk, q1k, q2k = (60.0, 0.0, 0.0, 4.8), (80.0, 0.0, 2.0, 0.3), (80.0, 0.2, 2.3, 0.3)
    wk = vsum(q1k, q2k)
    gen = (vsum(bk, q1k, q2k), bk, wk, q1k, q2k)
    fjet_kin = (wk[0], wk[1], wk[2], 80.0)                           # fat jet on the W, W-like mass
    assert abs(vsum(bk, fjet_kin)[3] - M.TOP_MASS) < M.TOP_MASS_WINDOW
    assert score(M.SemiResolvedQQ_top, momenta, gen, sr_cand(momenta, bk, fjet_kin)) == pytest.approx(0.0, abs=1e-6)
    bad_mass = sr_cand(momenta, bk, (wk[0], wk[1], wk[2], 150.0))    # fat jet mass outside the W window
    assert score(M.SemiResolvedQQ_top, momenta, gen, bad_mass) == M.DR_FILL_VALUE
    bad_b = sr_cand(momenta, (60.0, 0.0, 0.6, 4.8), fjet_kin)        # jet 0.6 from the b quark (> JET_DR)
    assert score(M.SemiResolvedQQ_top, momenta, gen, bad_b) == M.DR_FILL_VALUE


def test_semi_resolved_bq_takes_the_better_of_both_orientations(momenta):
    # BQ1: fat jet holds b and q1, jet on q2. BQ2: fat jet holds b and q2, jet on q1.
    bk, q1k, q2k = (90.0, 0.0, 0.0, 4.8), (60.0, 0.3, 0.2, 0.3), (60.0, 0.0, 2.5, 0.3)
    gen = (vsum(bk, q1k, q2k), bk, vsum(q1k, q2k), q1k, q2k)
    fj = (150.0, 0.15, 0.1, 60.0)                                    # between b and q1
    assert abs(vsum(fj, q2k)[3] - M.TOP_MASS) < M.TOP_MASS_WINDOW
    cand = sr_cand(momenta, q2k, fj)
    bq1, bq2 = score(M.SemiResolvedBQ1_top, momenta, gen, cand), score(M.SemiResolvedBQ2_top, momenta, gen, cand)
    assert bq1 != M.DR_FILL_VALUE and bq2 == M.DR_FILL_VALUE          # only the BQ1 orientation fits
    combined = score(M.SemiResolvedBQ_top, momenta, gen, cand)
    assert combined == pytest.approx(min(bq1, bq2))
    # relabelling the two W quarks must not change the combined score
    swapped = (gen[0], gen[1], gen[2], q2k, q1k)
    assert score(M.SemiResolvedBQ_top, momenta, swapped, cand) == pytest.approx(combined)


# ------------------------------------------------------------- fully boosted
def test_fully_boosted_gates(momenta):
    bk, q1k, q2k = (150.0, 0.0, 0.0, 4.8), (150.0, 0.2, 0.2, 0.3), (150.0, -0.2, -0.2, 0.3)
    topk = vsum(bk, q1k, q2k)
    gen = (topk, bk, vsum(q1k, q2k), q1k, q2k)
    ok = (400.0, 0.0, 0.0, 172.0)
    assert score(M.FullyBoosted_top, momenta, gen, fb_cand(momenta, ok)) == pytest.approx(vobj(ok).deltaR(vobj(topk)), abs=1e-6)
    assert score(M.FullyBoosted_top, momenta, gen, fb_cand(momenta, (300.0, 0.0, 0.0, 172.0))) == M.DR_FILL_VALUE  # pT <= FB_PTCUT
    assert score(M.FullyBoosted_top, momenta, gen, fb_cand(momenta, (400.0, 0.0, 0.0, 90.0))) == M.DR_FILL_VALUE   # mass window
    assert score(M.FullyBoosted_top, momenta, gen, fb_cand(momenta, (400.0, 0.0, 1.0, 172.0))) == M.DR_FILL_VALUE  # outside cone


# ---------------------------------------------------------------- overlaps
def test_semi_resolved_overlap_predicate(momenta):
    rec = lambda cand: cand[0][0]
    a = rec(sr_cand(momenta, (50.0, 0.0, 0.0, 5.0), (200.0, 0.0, 2.0, 80.0)))
    same_jet = rec(sr_cand(momenta, (50.0, 0.0, 0.3, 5.0), (200.0, 1.5, -2.5, 80.0)))      # jets 0.3 apart (< JET_DR)
    same_fjet = rec(sr_cand(momenta, (50.0, 1.5, -2.5, 5.0), (200.0, 0.0, 2.5, 80.0)))     # fat jets 0.5 apart (< FJET_DR)
    jet_in_fjet = rec(sr_cand(momenta, (50.0, 0.0, 2.5, 5.0), (200.0, 1.5, -2.5, 80.0)))   # B's jet inside A's fat jet
    disjoint = rec(sr_cand(momenta, (50.0, 1.5, -2.5, 5.0), (200.0, -1.5, -1.0, 80.0)))
    assert M.SemiResolved_overlap(a, same_jet) and M.SemiResolved_overlap(a, same_fjet) and M.SemiResolved_overlap(a, jet_in_fjet)
    assert not M.SemiResolved_overlap(a, disjoint)


def test_fully_boosted_overlap_predicate(momenta):
    a = fb_cand(momenta, (400.0, 0.0, 0.0, 172.0))[0][0]
    assert M.FullyBoosted_overlap(a, fb_cand(momenta, (400.0, 0.0, 0.7, 172.0))[0][0])
    assert not M.FullyBoosted_overlap(a, fb_cand(momenta, (400.0, 0.0, 0.9, 172.0))[0][0])


# ---------------------------------------------------------- fat-jet matcher
@pytest.mark.parametrize("order", [(0.3, 0.5), (0.5, 0.3)])
def test_match_fjet_to_jet_picks_the_closest_cone(momenta, order):
    jets = momenta([(40.0, 0.0, 0.0, 5.0)])
    fjets = momenta([(200.0, 0.0, order[0], 80.0), (200.0, 0.0, order[1], 80.0)])
    idx, dr = M.match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    assert ak.to_list(idx.snapshot())[0] == [int(np.argmin(order))]
    assert ak.to_list(dr.snapshot())[0] == [pytest.approx(min(order))]


# ---------------------------------------------------------- reconstruct_top
def test_reconstruct_top_returns_fill_when_nothing_qualifies(momenta, fr_gen):
    tops, bq, ws, q1a, q2a = (momenta([k]) for k in fr_gen)
    jets = momenta([(50.0, 2.0, -2.0, 5.0), (50.0, -2.0, 2.0, 5.0), (50.0, 2.0, 2.0, 5.0)])   # nowhere near the quarks
    cands = ak.zip({"bjet": jets[:, [0]], "q1jet": jets[:, [1]], "q2jet": jets[:, [2]]})
    picked = M.reconstruct_top(tops, bq, ws, q1a, q2a, cands, M.FullyResolved_top, M.FullyResolved_overlap, ak.ArrayBuilder())
    assert ak.to_list(picked.snapshot()) == [[M.NOJET_FILL_VALUE]]


def test_reconstruct_top_prefers_the_lower_score(momenta, fr_gen):
    tops, bq, ws, q1a, q2a = (momenta([k]) for k in fr_gen)
    b_off = (B[0], B[1], B[2] + 0.3, B[3])                           # exact candidate scores 0, shifted one 0.3
    cands = ak.concatenate([fr_cand(momenta, b_off, Q1, Q2), fr_cand(momenta, B, Q1, Q2)], axis=1)
    picked = M.reconstruct_top(tops, bq, ws, q1a, q2a, cands, M.FullyResolved_top, M.FullyResolved_overlap, ak.ArrayBuilder())
    assert ak.to_list(picked.snapshot()) == [[1]]
