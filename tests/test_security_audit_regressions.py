"""Harmless generated fixtures for local file and publication boundaries."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_fixed_temporary_symlink_cannot_redirect_json(tmp_path):
    from mindwatch.research.benchmarks import _json
    repo = tmp_path / "synthetic_repo"
    (repo / ".git").mkdir(parents=True)
    sentinel = repo / "untouched.py"
    sentinel.write_text("# untouched\n")
    private = tmp_path / "private"
    private.mkdir()
    (private / "selection.json.tmp").symlink_to(sentinel)
    _json(private / "selection.json", {"status": "synthetic"})
    assert sentinel.read_text() == "# untouched\n"
    assert json.loads((private / "selection.json").read_text())["status"] == "synthetic"


def test_atomic_failure_preserves_previous_artifact(tmp_path):
    from mindwatch.research.benchmarks import _json
    path = tmp_path / "selection.json"
    path.write_text('{"old": true}')
    with pytest.raises(ValueError):
        _json(path, {"invalid": float("nan")})
    assert path.read_text() == '{"old": true}'
    assert not list(tmp_path.glob(".write-*"))


def test_checkpoint_is_private_while_writing_and_rejects_symlink(tmp_path, monkeypatch):
    from mindwatch.research.benchmarks import _checkpoint
    import joblib
    path = tmp_path / "model.joblib"
    path.write_bytes(b"old")
    path.chmod(0o644)
    real_dump = joblib.dump

    def checked_dump(model, destination, **kwargs):
        assert Path(destination).stat().st_mode & 0o777 == 0o600
        assert path.read_bytes() == b"old"
        return real_dump(model, destination, **kwargs)

    monkeypatch.setattr(joblib, "dump", checked_dump)
    _checkpoint(path, {"synthetic": True})
    assert path.stat().st_mode & 0o777 == 0o600
    alias = tmp_path / "alias.joblib"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        _checkpoint(alias, {})


def test_cli_does_not_print_private_exception_payload(tmp_path, monkeypatch, capsys):
    import mindwatch.cli as cli
    marker = "synthetic-confidential-cell"
    def bad_input(_):
        raise ValueError(marker)
    monkeypatch.setattr(cli, "read_table", bad_input)
    before = os.umask(0o077)
    try:
        code = cli.main(["train", "--train-visits", "dummy.csv",
                         "--selection-visits", "dummy.csv", "--calibration-visits", "dummy.csv",
                         "--ema", "dummy.csv", "--output", str(tmp_path / "private")])
    finally:
        os.umask(before)
    captured = capsys.readouterr()
    assert code == 1
    assert marker not in captured.out + captured.err
    assert "ValueError" in captured.err


def test_push_checks_commits_even_if_known_to_another_remote(tmp_path, monkeypatch, capsys):
    source = Path(__file__).resolve().parents[1] / "tools" / "check_code_only.py"
    spec = importlib.util.spec_from_file_location("synthetic_code_check", source)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args]).decode().strip()
    git("init", "-q")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (tmp_path / "synthetic.csv").write_text("synthetic-only\n")
    git("add", "synthetic.csv")
    git("commit", "-qm", "Synthetic test commit")
    head = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/private/main", head)
    # A clean tip must not hide an intermediate forbidden blob either.
    git("rm", "synthetic.csv")
    (tmp_path / "README.md").write_text("Synthetic code fixture\n")
    git("add", "README.md")
    git("commit", "-qm", "Synthetic clean tip")
    tip = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/private/main", tip)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"refs/heads/main {tip} refs/heads/main {'0'*40}\n"))
    assert checker.main(["--push"]) == 1
    assert "synthetic.csv" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["run_full_pipeline", "run_tabular_models", "train_tft",
                                 "train_tabular_dl", "tune_tabular_models", "run_optuna_batch",
                                 "evaluate_tabular_strategy"])
def test_legacy_training_stops_before_optional_imports_or_input_access(tmp_path, name):
    script = Path(__file__).resolve().parents[1] / "src" / (name + ".py")
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "historical training driver is retired" in result.stderr
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_legacy_subprocess_log_does_not_echo_storage_credentials(monkeypatch, capsys):
    import run_optuna_batch as batch
    marker = "synthetic-password"
    monkeypatch.setattr(batch.subprocess, "run", lambda args: subprocess.CompletedProcess(args, 1))
    assert batch.run_command(["python", "synthetic.py", "--storage", f"postgresql://user:{marker}@localhost/db"]) == 1
    output = capsys.readouterr()
    assert marker not in output.out + output.err


@pytest.mark.parametrize("kind", ["commit", "tag"])
def test_push_scans_metadata_without_echoing_matching_content(tmp_path, monkeypatch, capsys, kind):
    source = Path(__file__).resolve().parents[1] / "tools" / "check_code_only.py"
    spec = importlib.util.spec_from_file_location("synthetic_metadata_check", source)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args]).decode().strip()
    git("init", "-q")
    git("config", "user.name", "Synthetic Test")
    git("config", "user.email", "synthetic@example.invalid")
    (tmp_path / "README.md").write_text("Synthetic test fixture\n")
    git("add", "README.md")
    marker = "ghp_" + "0" * 40
    git("commit", "-qm", marker if kind == "commit" else "Safe synthetic message")
    ref = "HEAD"
    if kind == "tag":
        git("tag", "-a", "synthetic-tag", "-m", marker)
        ref = "refs/tags/synthetic-tag"
    oid = git("rev-parse", ref)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{ref} {oid} {ref} {'0'*40}\n"))
    assert checker.main(["--push"]) == 1
    output = capsys.readouterr()
    assert marker not in output.out + output.err
    assert "metadata" in output.err


@pytest.mark.parametrize("module,args", [
    ("mindwatch.cli", ["train", "--train-visits", "x.csv", "--selection-visits", "x.csv",
                       "--calibration-visits", "x.csv", "--ema", "x.csv", "--output", "x"]),
    ("mindwatch.research.cli", ["prepare", "--output", "x"]),
])
def test_invalid_cli_arguments_are_not_echoed(module, args):
    marker = "synthetic-confidential-argument"
    run = subprocess.run([sys.executable, "-m", module, *args, "--typo", marker], capture_output=True, text=True)
    assert run.returncode == 2
    assert marker not in run.stdout + run.stderr
    assert "Invalid command line" in run.stderr
