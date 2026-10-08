"""LPCVanillaSubmitter writes one executable and one submit file per job
without touching git, voms or condor (all shell-outs are stubbed)."""
import glob
import os
import subprocess
import types

import pytest

from src.data.delphes import condor_conversion as CC


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: types.SimpleNamespace(communicate=lambda: (b"/repo\n", None)))
    monkeypatch.setattr(subprocess, "getoutput", lambda cmd: "20260101_000000")
    monkeypatch.setattr(subprocess, "getstatusoutput", lambda cmd: (0, "/tmp/x509up_u1234"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    chmod_calls = []
    monkeypatch.setattr(os, "system", lambda cmd: chmod_calls.append(cmd) or 0)
    return tmp_path, chmod_calls


def test_job_files_cover_every_input_group(stubbed):
    tmp_path, chmod_calls = stubbed
    CC.LPCVanillaSubmitter([["/d/f1.root", "/d/f2.root"], ["/d/f3.root"]], "/out/tttt_training.h5", memory="8GB")
    files = sorted(glob.glob(str(tmp_path / ".condor_preprocess" / "20260101_000000" / "**" / "*"), recursive=True))
    texts = {f: open(f).read() for f in files if os.path.isfile(f)}
    executables = [t for t in texts.values() if t.startswith("#!/bin/bash")]
    submits = [t for t in texts.values() if "request_memory" in t]
    assert executables and submits, files
    joined = "\n".join(executables)
    assert "convert_to_h5.py" in joined and "--out-file" in joined
    assert "f1.root /d/f2.root" in joined and "f3.root" in joined
    assert "if [ $1 -eq 0 ]" in joined and "if [ $1 -eq 1 ]" in joined
    assert all("request_memory = 8GB" in s and "arguments = $(ProcId)" in s for s in submits)
    assert chmod_calls and all(c.startswith("chmod 775") for c in chmod_calls)
