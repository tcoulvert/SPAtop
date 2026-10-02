"""Round-trip tests for the h5 post-processing utilities: merge, split, concat."""
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from src.data.delphes.merge_dataset import merge_h5_files, merge_multiple_h5_files
from src.data.delphes.split_dataset import split_h5_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, start, n):
    """n events whose 'uid' column identifies them across files."""
    with h5py.File(path, "w") as f:
        f.create_dataset("INPUTS/Jets/btag", data=np.zeros((n, 4), bool))
        f.create_dataset("INPUTS/Jets/pt", data=np.arange(start, start + n, dtype="float32")[:, None] * np.ones((1, 4), "float32"))
        f.create_dataset("TARGETS/FRt1/MASK", data=np.ones(n, bool))
        f.create_dataset("TARGETS/FRt1/b", data=np.arange(start, start + n, dtype="int32"))


def uids(path, key="TARGETS/FRt1/b"):
    with h5py.File(path) as f:
        return f[key][:].tolist()


def test_merge_two_files_preserves_every_event(tmp_path):
    a, b, out = tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "ab.h5"
    write(a, 0, 5); write(b, 100, 3)
    merge_h5_files(str(a), str(b), str(out))
    assert uids(out) == list(range(5)) + [100, 101, 102]
    with h5py.File(out) as f:
        assert f["INPUTS/Jets/pt"].shape == (8, 4)


def test_merge_with_shuffle_is_a_seeded_permutation(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"; write(a, 0, 5); write(b, 100, 3)
    merge_h5_files(str(a), str(b), str(tmp_path / "s1.h5"), shuffle=True, seed=7)
    merge_h5_files(str(a), str(b), str(tmp_path / "s2.h5"), shuffle=True, seed=7)
    s1 = uids(tmp_path / "s1.h5")
    assert sorted(s1) == list(range(5)) + [100, 101, 102] and s1 == uids(tmp_path / "s2.h5")
    with h5py.File(tmp_path / "s1.h5") as f:                         # rows stay aligned across datasets
        assert f["INPUTS/Jets/pt"][:, 0].astype(int).tolist() == s1


def test_merge_multiple_requires_two_files_and_counts_events(tmp_path):
    files = []
    for i, n in enumerate((2, 3, 4)):
        p = tmp_path / f"f{i}.h5"; write(p, 10 * i, n); files.append(str(p))
    with pytest.raises(ValueError):
        merge_multiple_h5_files(files[:1], str(tmp_path / "x.h5"))
    merge_multiple_h5_files(files, str(tmp_path / "m.h5"))
    assert sorted(uids(tmp_path / "m.h5")) == [0, 1, 10, 11, 12, 20, 21, 22, 23]


def test_split_is_disjoint_covering_and_seeded(tmp_path):
    src = tmp_path / "data.h5"; write(src, 0, 10)
    split_h5_file(str(src), train_ratio=0.6, output_dir=str(tmp_path / "out"), seed=3)
    tr, va = uids(tmp_path / "out" / "data_train.h5"), uids(tmp_path / "out" / "data_val.h5")
    assert len(tr) == 6 and len(va) == 4 and sorted(tr + va) == list(range(10))
    split_h5_file(str(src), train_ratio=0.6, output_dir=str(tmp_path / "out2"), seed=3)
    assert uids(tmp_path / "out2" / "data_train.h5") == tr


def test_concat_script_concatenates_matching_files(tmp_path):
    a, b, out = tmp_path / "a.h5", tmp_path / "b.h5", tmp_path / "out.h5"
    write(a, 0, 2); write(b, 50, 3)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "data", "delphes", "concat_h5.py"), str(out), str(a), str(b)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert uids(out) == [0, 1, 50, 51, 52]
