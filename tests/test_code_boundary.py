import importlib.util
from pathlib import Path

import pytest

LOCATION = Path(__file__).resolve().parents[1]/"tools"/"check_code_only.py"
spec = importlib.util.spec_from_file_location("code_boundary", LOCATION)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("path", ["data/records.csv", "weights.joblib", "reports/analysis.md", "run.log",
                                 "src/data.parquet", "tests/fixture.json", "notebook.ipynb", "../escape.py"])
def test_noncode_paths_rejected(path):
    assert checker.inspect_content(path, b"innocent-looking text")


def test_disguised_binary_symlink_private_path_credential_rejected():
    assert checker.inspect_content("src/model.py", b"binary\x00stuff")
    assert checker.inspect_content("src/link.py", b"print(1)", "120000")
    private = "/"+"home"+"/synthetic_user/records"
    assert checker.inspect_content("src/path.py", private.encode())
    token = "ghp_"+"x"*40
    assert checker.inspect_content("src/secret.py", token.encode())
    assert not checker.inspect_content("src/module.py", b"def f():\n    return 1\n")


def test_legacy_helpers(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(LOCATION.parents[1]/"src"))
    from utils import get_data_paths, load_config
    config = tmp_path/"private.yaml"
    config.write_text("input_data_dir: inputs\nlabel_data_dir: labels\nvoice_data_dir: voice\n", encoding="utf-8")
    monkeypatch.setenv("MINDWATCH_CONFIG", str(config))
    assert set(get_data_paths(load_config())) == {"input", "label", "voice"}


def test_checker_reads_index_not_working_tree(tmp_path, monkeypatch, capsys):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path/"src").mkdir()
    source = tmp_path/"src"/"example.py"
    source.write_text("print('safe synthetic code')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src/example.py"], check=True)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    assert checker.main(["--staged"]) == 0
    forbidden = tmp_path/"records.csv"
    forbidden.write_text("synthetic-only\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "records.csv"], check=True)
    forbidden.unlink()
    assert checker.main(["--staged"]) == 1
    assert "records.csv" in capsys.readouterr().err


def test_legacy_plot_helper_uses_private_output(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from utils import generate_plots
    generate_plots([0, 1, 0, 1], [.1, .8, .2, .9], "synthetic", tmp_path, "check")
    assert len(list(tmp_path.glob("*.png"))) == 2
