"""Shared fixtures for the converter test suite.

The real-data fixture is 24 events of pp -> tttt (all-hadronic) Delphes output,
sliced from /data/spatop/tttt_15M/root/tttt_hadronic_0.root (entries listed in
tests/fixtures/README.md), restricted to the 36 branches the converter reads,
and chosen so every event passes event selection with four hadronic tops.
"""
import os
import sys

import awkward as ak
import numpy as np
import pytest
import vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

vector.register_awkward()

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "tttt_delphes_24.parquet")
N_TOPS = 4


@pytest.fixture(scope="session")
def arrays():
    return ak.from_parquet(FIXTURE)


def convert(arrays, n_tops=N_TOPS, n_targets=N_TOPS, min_valid_targets=1):
    """get_datasets with the tagger-emulation RNG reset to the converter's own seed.

    The converter draws its emulated tagger decisions from a module-level stream
    (RNG = default_rng(21)), so a fresh process is reproducible but a second
    conversion in the same process is not. Resetting here keeps every test
    independent of the order pytest runs them in."""
    from src.data.delphes import convert_to_h5 as C
    C.RNG = np.random.default_rng(seed=21)
    return C.get_datasets(arrays, n_tops, n_targets, min_valid_targets)


@pytest.fixture(scope="session")
def converted(arrays):
    """Converter output on the fixture: n_tops 4, n_targets 4, min_valid_targets 1."""
    return convert(arrays)


@pytest.fixture
def momenta():
    """list of (pt, eta, phi, mass) for ONE event -> jagged Momentum4D array [1, n]."""
    def build(kin):
        cols = {k: [np.array([x[i] for x in kin], dtype=float)] for i, k in enumerate(("pt", "eta", "phi", "mass"))}
        return ak.zip(cols, with_name="Momentum4D")
    return build


@pytest.fixture
def record(momenta):
    """single Momentum4D record for njit scoring functions."""
    return lambda pt, eta, phi, mass: momenta([(pt, eta, phi, mass)])[0][0]
