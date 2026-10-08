"""Regression tests for the Delphes -> h5 converter fixes.

Each test pins one defect found in a differential run of the converter on a
real tttt Delphes file (see the commit message), so a reintroduction fails
loudly instead of silently changing the training labels.
"""
import inspect

import numba as nb

import awkward as ak
import numpy as np
import pytest
import vector

vector.register_awkward()

from src.data.delphes import convert_to_h5 as C
from src.data.delphes import matching as M


# ----------------------------------------------------------------- helpers
def particles(rows):
    """rows: list of (pid, status, d1, d2); returns one event as an ak record."""
    pid, status, d1, d2 = (np.array([r[i] for r in rows]) for i in range(4))
    return ak.Array({"idx": [np.arange(len(rows))], "pid": [pid], "status": [status], "d1": [d1], "d2": [d2]})


def last_copy(parts, start):
    inter = parts[ak.local_index(parts.pid) == start]
    return int(ak.flatten(C.final_particle(inter, parts).idx)[0])


def momenta(kin):
    """kin: list of (pt, eta, phi, mass) for one event -> Momentum4D records."""
    pt, eta, phi, mass = (np.array([k[i] for k in kin], dtype=float) for i in range(4))
    return ak.zip({"pt": [pt], "eta": [eta], "phi": [phi], "mass": [mass]}, with_name="Momentum4D")


# ------------------------------------------------- fix 4: decay-chain walker
def test_walker_follows_copy_in_second_daughter_slot():
    # W(0) -> gamma(1) + W(2); W(2) -> u(3) dbar(4).  The copy sits in d2.
    parts = particles([(24, 22, 1, 2), (22, 1, -1, -1), (24, 51, 3, 4), (2, 23, -1, -1), (-1, 23, -1, -1)])
    assert last_copy(parts, 0) == 2


def test_walker_ignores_status_ordering():
    # W(0, status 52) -> W(1, status 52) -> W(2, status 51) -> quarks.  Statuses never increase.
    parts = particles([(24, 52, 1, -1), (24, 52, 2, -1), (24, 51, 3, 4), (1, 23, -1, -1), (-2, 23, -1, -1)])
    assert last_copy(parts, 0) == 2


def test_walker_treats_minus_one_as_no_daughter():
    # the final particle has d1 = d2 = -1; numpy would wrap -1 to the LAST particle (a W!) if unguarded
    parts = particles([(24, 22, 1, -1), (24, 51, -1, -1), (24, 1, 0, 0)])
    assert last_copy(parts, 0) == 1


# --------------------------------------------- fix 3: symmetric FR overlap
@pytest.mark.parametrize("role_a, role_b", [("q1jet", "bjet"), ("q2jet", "bjet"), ("q2jet", "q1jet")])
def test_fully_resolved_overlap_catches_every_role_pairing(role_a, role_b):
    # candidate A uses jet X as role_a; candidate B uses the SAME jet as role_b.
    # Before the fix only six of the nine role pairings were checked.
    far = lambda i: (50.0, 0.0, 0.6 * i, 5.0)
    x = (60.0, 1.0, 1.0, 5.0)
    a = {"bjet": far(1), "q1jet": far(2), "q2jet": far(3)}; a[role_a] = x
    b = {"bjet": far(4), "q1jet": far(5), "q2jet": far(6)}; b[role_b] = x
    cand = lambda d: ak.zip({k: momenta([v]) for k, v in d.items()})[0][0]
    assert bool(M.FullyResolved_overlap(cand(a), cand(b)))
    assert bool(M.FullyResolved_overlap(cand(b), cand(a)))


# ------------------------------------------- fix 7: deltaRfj encoding
def test_deltaRfj_is_the_real_distance_outside_every_cone():
    jets = momenta([(40.0, 0.0, 0.0, 5.0), (40.0, 0.0, 2.0, 5.0)])
    fjets = momenta([(200.0, 0.0, 0.3, 80.0)])
    idx, dr = M.match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    idx, dr = ak.to_list(idx.snapshot())[0], ak.to_list(dr.snapshot())[0]
    assert idx == [0, M.NOJET_FILL_VALUE]
    assert dr[0] == pytest.approx(0.3) and dr[1] == pytest.approx(1.7)


def test_deltaRfj_without_fat_jets_is_the_pad():
    jets = momenta([(40.0, 0.0, 0.0, 5.0)])
    fjets = ak.zip({k: [np.array([], dtype=float)] for k in ("pt", "eta", "phi", "mass")}, with_name="Momentum4D")
    idx, dr = M.match_fjet_to_jet(fjets, jets, ak.ArrayBuilder(), ak.ArrayBuilder())
    assert ak.to_list(idx.snapshot())[0] == [M.NOJET_FILL_VALUE]
    assert ak.to_list(dr.snapshot())[0] == [pytest.approx(M.NOFJET_DR_FILL_VALUE)]


# ------------------------------------------ fix 2: overlap exclusion works
@nb.njit
def _never_overlaps(a, b):
    return False


def test_second_top_cannot_reuse_first_tops_jets():
    """Two hadronic tops. Jet 2 lies exactly on top 1's q2 quark and only
    0.25 in phi from top 2's q1 quark; jet 5 lies 0.45 from that quark on the
    other side, so jets 2 and 5 are 0.70 apart (no jet-jet overlap).
    Top 1 is scored first and takes jet 2. Top 2 then scores jet 2 better
    than jet 5, so without working exclusion it reuses jet 2."""
    q = {'b1': (110, 0.7, 0.9, 4.8), 'q11': (70, -0.7, 0.9, 0.3), 'q12': (70, 0.0, 1.5, 0.3), 'b2': (110, -0.7, 2.35, 4.8), 'q21': (70, 0.0, 1.75, 0.3), 'q22': (70, 0.7, 2.35, 0.3)}
    jets = momenta([(110, 0.7, 0.9, 4.8), (70, -0.7, 0.9, 0.3), (70, 0.0, 1.5, 0.3), (110, -0.7, 2.35, 4.8), (70, 0.7, 2.35, 0.3), (70, 0.0, 2.2, 0.3)])
    pair = lambda x, y: ak.concatenate([momenta([q[x]]), momenta([q[y]])], axis=1)
    bq, q1, q2 = pair("b1", "b2"), pair("q11", "q21"), pair("q12", "q22")
    ws = q1 + q2
    tops = bq + ws
    i0, i1, i2 = ak.unzip(ak.argcartesian([jets, jets, jets], axis=1))
    keep = (i0 != i1) & (i0 != i2) & (i1 != i2)
    i0, i1, i2 = i0[keep], i1[keep], i2[keep]
    cands = ak.zip({"bjet": jets[i0], "q1jet": jets[i1], "q2jet": jets[i2]})

    def assign(overlap_check):
        picked = M.reconstruct_top(tops, bq, ws, q1, q2, cands, M.FullyResolved_top, overlap_check, ak.ArrayBuilder())
        return [(int(i0[0][p]), int(i1[0][p]), int(i2[0][p])) if p >= 0 else None for p in ak.to_list(picked.snapshot())[0]]

    # positive control: with exclusion disabled the second top reuses jet 2
    assert assign(_never_overlaps) == [(0, 1, 2), (3, 2, 4)]
    # with the fix, the second top is forced onto the unused jet 5
    assert assign(M.FullyResolved_overlap) == [(0, 1, 2), (3, 5, 4)]


def test_matched_overlap_sees_recorded_assignments():
    far = lambda i: (50.0, 0.0, 0.6 * i, 5.0)
    cands = ak.zip({"bjet": momenta([far(1), far(4)]), "q1jet": momenta([far(2), far(5)]), "q2jet": momenta([far(3), far(1)])})[0]
    from numba.typed import List
    from numba import types
    matched = List.empty_list(types.int64); matched.append(0)
    assert bool(M.matched_overlap(0, matched, cands, M.FullyResolved_overlap))      # already assigned
    assert bool(M.matched_overlap(1, matched, cands, M.FullyResolved_overlap))      # shares jet far(1) with candidate 0
    empty = List.empty_list(types.int64)
    assert not bool(M.matched_overlap(1, empty, cands, M.FullyResolved_overlap))


# ------------------------------------------------- fix 1 / fix 5: CLI surface
def test_cli_has_no_multip_and_n_targets_defaults_to_n_tops():
    assert "multip" not in inspect.signature(C.main.callback).parameters
    n_targets = next(p for p in C.main.params if p.name == "n_targets")
    assert n_targets.default is None
