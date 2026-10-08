"""Integration tests of get_datasets on real Delphes events (tests/fixtures):
the output contract every downstream consumer relies on, plus the physics
invariants that the July/September bugs violated."""
import awkward as ak
import numpy as np
import pytest

from src.data.delphes import convert_to_h5 as C
from src.data.delphes import matching as M
from tests.conftest import convert

N_TOPS = 4
JET_FEATURES = ["MASK", "pt", "eta", "phi", "sinphi", "cosphi", "mass", "btag", "flavor", "matchedfj", "deltaRfj"]
FJ_FEATURES = ["MASK", "fj_pt", "fj_eta", "fj_phi", "fj_sinphi", "fj_cosphi", "fj_mass", "fj_sdmass", "fj_Ttag", "fj_Wtag",
               "fj_tau21", "fj_tau32", "fj_charge", "fj_ehadovereem", "fj_neutralenergyfrac", "fj_chargedenergyfrac",
               "fj_nneutral", "fj_ncharged"]
TOPOLOGIES = {"FR": ["b", "q1", "q2"], "SRqq": ["b", "qq"], "SRbq": ["q", "bq"], "FB": ["bqq"]}
FATJET_FIELDS = {"qq", "bq", "bqq"}


def present(converted, topo, i):
    return converted[f"TARGETS/{topo}t{i}/MASK"].astype(bool)


def test_dataset_contract(converted):
    expected = {f"INPUTS/Jets/{f}" for f in JET_FEATURES} | {f"INPUTS/BoostedJets/{f}" for f in FJ_FEATURES}
    for topo, fields in TOPOLOGIES.items():
        for i in range(1, N_TOPS + 1):
            expected |= {f"TARGETS/{topo}t{i}/{f}" for f in ["MASK", "pt"] + fields}
    assert set(converted) == expected
    n = converted["INPUTS/Jets/pt"].shape[0]
    assert n > 0 and all(v.shape[0] == n for v in converted.values())
    assert converted["INPUTS/Jets/pt"].shape[1] == 3 * N_TOPS + 4
    assert converted["INPUTS/BoostedJets/fj_pt"].shape[1] == N_TOPS + 1


def test_dtypes(converted):
    assert converted["INPUTS/Jets/MASK"].dtype == bool and converted["INPUTS/BoostedJets/MASK"].dtype == bool
    assert converted["INPUTS/Jets/deltaRfj"].dtype == np.float32           # was int32 before 6ede399
    assert converted["INPUTS/Jets/matchedfj"].dtype == np.int32
    assert converted["INPUTS/BoostedJets/fj_Ttag"].dtype == bool and converted["INPUTS/BoostedJets/fj_Wtag"].dtype == bool
    for topo, fields in TOPOLOGIES.items():
        for f in fields:
            assert np.issubdtype(converted[f"TARGETS/{topo}t1/{f}"].dtype, np.integer)
        assert converted[f"TARGETS/{topo}t1/MASK"].dtype == bool


def test_inputs_are_pt_sorted_masked_and_selected(converted):
    for coll, pre in (("Jets", ""), ("BoostedJets", "fj_")):
        m, pt = converted[f"INPUTS/{coll}/MASK"], converted[f"INPUTS/{coll}/{pre}pt"]
        assert np.array_equal(m, pt > 0)
        assert (pt[m] >= (C.MIN_JET_PT if coll == "Jets" else C.MIN_FJET_PT)).all()
        assert (np.diff(pt, axis=1) <= 0).all()                          # descending, zero padding at the end
        assert np.allclose(converted[f"INPUTS/{coll}/{pre}sinphi"][m], np.sin(converted[f"INPUTS/{coll}/{pre}phi"][m]), atol=1e-5)
        assert np.allclose(converted[f"INPUTS/{coll}/{pre}cosphi"][m], np.cos(converted[f"INPUTS/{coll}/{pre}phi"][m]), atol=1e-5)
    assert (converted["INPUTS/Jets/MASK"].sum(1) >= 3 * N_TOPS).all()        # event selection: enough jets


def test_target_masks_indices_and_pt_are_consistent(converted):
    n_jets, n_fjets = converted["INPUTS/Jets/MASK"].sum(1), converted["INPUTS/BoostedJets/MASK"].sum(1)
    for topo, fields in TOPOLOGIES.items():
        for i in range(1, N_TOPS + 1):
            m = present(converted, topo, i)
            for f in fields:
                idx = converted[f"TARGETS/{topo}t{i}/{f}"]
                assert np.array_equal(m, idx != M.NOJET_FILL_VALUE)
                limit = n_fjets if f in FATJET_FIELDS else n_jets
                assert (idx[m] < limit[m]).all() and (idx[m] >= 0).all()
            assert (converted[f"TARGETS/{topo}t{i}/pt"][m] > 0).all()
    assert any(present(converted, topo, i).any() for topo in TOPOLOGIES for i in range(1, N_TOPS + 1))


def test_no_object_is_claimed_by_two_tops_of_the_same_topology(converted):
    n = converted["INPUTS/Jets/pt"].shape[0]
    for topo, fields in TOPOLOGIES.items():
        for e in range(n):
            objs = [("fj" if f in FATJET_FIELDS else "j", int(converted[f"TARGETS/{topo}t{i}/{f}"][e]))
                    for i in range(1, N_TOPS + 1) if present(converted, topo, i)[e] for f in fields]
            assert len(objs) == len(set(objs)), (topo, e, objs)


def test_matchedfj_and_deltaRfj_follow_their_definitions(converted):
    jm, fm = converted["INPUTS/Jets/MASK"], converted["INPUTS/BoostedJets/MASK"]
    je, jp = converted["INPUTS/Jets/eta"].astype(float), converted["INPUTS/Jets/phi"].astype(float)
    fe, fp = converted["INPUTS/BoostedJets/fj_eta"].astype(float), converted["INPUTS/BoostedJets/fj_phi"].astype(float)
    dphi = (jp[:, :, None] - fp[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    dr = np.where(fm[:, None, :], np.sqrt((je[:, :, None] - fe[:, None, :]) ** 2 + dphi ** 2), np.inf)
    inside = dr < M.FJET_DR
    closest = np.where(inside.any(2), np.argmin(np.where(inside, dr, np.inf), 2), M.NOJET_FILL_VALUE)
    expected_dr = np.where(fm.any(1)[:, None], dr.min(2), M.NOFJET_DR_FILL_VALUE)
    assert np.array_equal(converted["INPUTS/Jets/matchedfj"][jm], closest[jm])
    assert np.allclose(converted["INPUTS/Jets/deltaRfj"][jm], expected_dr[jm], atol=1e-5)


def test_tagger_emulation_only_on_valid_fat_jets_and_not_degenerate(converted):
    fm = converted["INPUTS/BoostedJets/MASK"]
    for tag in ("fj_Ttag", "fj_Wtag"):
        t = converted[f"INPUTS/BoostedJets/{tag}"]
        assert not t[~fm].any()
    # fat jets labelled as fully-boosted tops are T-tagged at ~83%: with the fixed RNG seed at least one is
    fb = np.zeros_like(fm)
    for i in range(1, N_TOPS + 1):
        m = present(converted, "FB", i); fb[np.where(m)[0], converted[f"TARGETS/FBt{i}/bqq"][m]] = True
    assert fb.any() and converted["INPUTS/BoostedJets/fj_Ttag"][fb].mean() > 0.5


def test_min_valid_targets_counts_tops_with_any_label(arrays, converted):
    # a "valid target" is a top reconstructed in at least one topology, not a label row
    valid_tops = sum(np.logical_or.reduce([present(converted, topo, i) for topo in TOPOLOGIES]).astype(int)
                     for i in range(1, N_TOPS + 1))
    assert (valid_tops >= 1).all()
    stricter = convert(arrays, min_valid_targets=2)
    assert stricter["INPUTS/Jets/pt"].shape[0] == int((valid_tops >= 2).sum())
    assert np.array_equal(stricter["INPUTS/Jets/pt"], converted["INPUTS/Jets/pt"][valid_tops >= 2])


def test_conversion_is_deterministic(arrays, converted):
    again = convert(arrays)
    assert set(again) == set(converted)
    for k in converted:
        assert np.array_equal(again[k], converted[k]), k
