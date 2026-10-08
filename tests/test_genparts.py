"""Gen-level particle identification on a hand-written Delphes-style record:
decaying tops found whichever daughter slot holds the b, copy chains followed,
hadronic vs leptonic W decays separated, tops sorted by pT, and the
exact-count event selection."""
import awkward as ak
import numpy as np
import pytest

from src.data.delphes import convert_to_h5 as C

# columns: pid, status, m1, m2, d1, d2, pt, eta, phi, mass
HADRONIC = [
    (6, 22, -1, -1, 2, 3, 200.0, 0.1, 0.1, 172.5),   # 0  t -> b(2) W+(3)
    (-6, 22, -1, -1, 5, 4, 150.0, -0.1, 3.0, 172.5), # 1  tbar -> W-(5) bbar(4)   (W listed first)
    (5, 23, 0, -1, -1, -1, 90.0, 0.2, 0.2, 4.8),     # 2  b
    (24, 22, 0, -1, 6, 7, 110.0, 0.0, -0.1, 80.4),   # 3  W+ -> gamma(6) + W+ copy(7)  (copy in d2)
    (-5, 23, 1, -1, -1, -1, 70.0, -0.2, 2.9, 4.8),   # 4  bbar
    (-24, 22, 1, -1, 8, 9, 80.0, 0.0, 3.1, 80.4),    # 5  W- -> d(8) ubar(9)
    (22, 1, 3, -1, -1, -1, 1.0, 0.0, 0.0, 0.0),      # 6  gamma
    (24, 51, 3, -1, 10, 11, 109.0, 0.0, -0.1, 80.4), # 7  W+ (last copy) -> u(10) dbar(11)
    (1, 23, 5, -1, -1, -1, 40.0, 0.3, 3.0, 0.0),     # 8  d
    (-2, 23, 5, -1, -1, -1, 40.0, -0.3, 3.2, 0.0),   # 9  ubar
    (2, 23, 7, -1, -1, -1, 55.0, 0.3, 0.0, 0.0),     # 10 u
    (-1, 23, 7, -1, -1, -1, 54.0, -0.3, -0.2, 0.0),  # 11 dbar
]
LEPTONIC = [row for row in HADRONIC]
LEPTONIC[8] = (13, 23, 5, -1, -1, -1, 40.0, 0.3, 3.0, 0.106)    # W- -> mu- nubar instead of d ubar
LEPTONIC[9] = (-14, 23, 5, -1, -1, -1, 40.0, -0.3, 3.2, 0.0)

NAMES = ["PID", "Status", "M1", "M2", "D1", "D2", "PT", "Eta", "Phi", "Mass"]


def make_arrays(*events):
    return ak.Array({f"Particle/Particle.{n}": [[row[i] for row in ev] for ev in events] for i, n in enumerate(NAMES)})


@pytest.fixture
def two_events():
    return make_arrays(HADRONIC, LEPTONIC)


def test_only_the_all_hadronic_event_passes_exact_count_selection(two_events):
    event_mask, *_ = C.get_genparts(two_events, 2, 2, np.array([True, True]))
    assert ak.to_list(event_mask) == [True, False]


def test_tops_sorted_by_pt_and_decay_products_identified_in_either_slot_order(two_events):
    _, _, tops, bs, ws, wd1, wd2 = C.get_genparts(two_events, 2, 2, np.array([True, True]))
    assert ak.to_list(tops.pt)[0] == [200.0, 150.0]               # the tbar lists W before b and is still found
    assert ak.to_list(bs.pid)[0] == [5, -5]
    assert ak.to_list(ws.pid)[0] == [24, -24]
    assert sorted(abs(p) for p in ak.to_list(wd1.pid)[0] + ak.to_list(wd2.pid)[0]) == [1, 1, 2, 2]


def test_w_copy_chain_is_followed_to_the_last_copy(two_events):
    # The W+ radiates a photon: its copy sits in the SECOND daughter slot (d2) behind the photon,
    # and the copy carries a status the old walker's status test did not accept. The returned W
    # must be that last copy (status 51), whose daughters are the quarks; the W- has no copy chain.
    _, _, _, _, ws, _, _ = C.get_genparts(two_events, 2, 2, np.array([True, True]))
    assert ak.to_list(ws.status)[0] == [51, 22]


def test_leptonic_top_is_not_counted_as_hadronic():
    arrays = make_arrays(LEPTONIC)
    event_mask, _, tops, *_ = C.get_genparts(arrays, 2, 1, np.array([True]))
    assert ak.to_list(event_mask) == [True]                       # exactly one hadronic top, as requested
    assert ak.to_list(tops.pid)[0] == [6]
