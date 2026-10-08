"""File I/O and the command-line surface of convert_to_h5."""
import logging

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from src.data.delphes import convert_to_h5 as C


def two_chunks():
    return {"INPUTS/Jets/pt": [np.arange(6, dtype="float32").reshape(2, 3), np.ones((3, 3), "float32")],
            "TARGETS/FRt1/MASK": [np.array([True, False]), np.array([True, True, False])]}


def test_save_file_h5_concatenates_chunks_and_reports_events(tmp_path):
    out = tmp_path / "x.h5"
    assert C.save_file(str(out), two_chunks()) == 5
    with h5py.File(out) as f:
        assert f["INPUTS/Jets/pt"].shape == (5, 3) and f["INPUTS/Jets/pt"].dtype == np.float32
        assert f["TARGETS/FRt1/MASK"][:].tolist() == [True, False, True, True, False]


def test_save_file_rejects_missing_or_unknown_extension(tmp_path):
    with pytest.raises(Exception, match="No filepath extension"):
        C.save_file(str(tmp_path / "noext"), two_chunks())
    with pytest.raises(NotImplementedError):
        C.save_file(str(tmp_path / "x.txt"), two_chunks())


def test_process_file_reports_failure_loudly_and_returns_sentinel(caplog):
    with caplog.at_level(logging.ERROR):
        assert C.process_file("/nonexistent/file.root", "out_training.h5", 0.8, 2, 2, 1) == 400
    assert any("Preprocessing failed" in r.message and r.levelno == logging.ERROR for r in caplog.records)


def test_cli_exits_nonzero_when_an_input_fails(tmp_path):
    result = CliRunner().invoke(C.main, ["/nonexistent/file.root", "--out-file", str(tmp_path / "o_training.h5"), "--n-tops", "2"])
    assert result.exit_code == 1
    assert "0 of 1 input files converted" in result.output


def test_cli_rejects_more_targets_than_tops(tmp_path):
    result = CliRunner().invoke(C.main, ["/nonexistent/file.root", "--out-file", str(tmp_path / "o.h5"), "--n-tops", "2", "--n-targets", "3"])
    assert result.exit_code != 0 and isinstance(result.exception, AssertionError)


def test_cli_surface():
    import inspect
    params = {p.name: p for p in C.main.params}
    assert "multip" not in inspect.signature(C.main.callback).parameters
    assert params["n_targets"].default is None and params["n_tops"].default == 2
    assert params["min_valid_targets"].default == 0
