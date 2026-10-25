"""Publication probes use invented content constructed only at test runtime."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest


LOCATION = Path(__file__).resolve().parents[1] / "tools" / "check_code_only.py"
spec = importlib.util.spec_from_file_location("publication_guard", LOCATION)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def inspect(source):
    return checker.inspect_content("src/example.py", source.encode())


@pytest.mark.parametrize("prefix,character,length", [
    ("hf_", "a", 34), ("github_pat_", "a", 82), ("ghp_", "a", 40),
    ("ghs_", "a", 40), ("sk-proj-", "a", 70), ("sk-ant-api03-", "a", 40),
    ("AKIA", "A", 16), ("ASIA", "B", 16), ("AIza", "a", 35),
    ("xoxb-", "a", 30), ("sk_live_", "a", 28), ("Bearer ", "a", 40),
])
def test_supported_credentials_are_blocked(prefix, character, length):
    token = prefix + character * length
    assert "possible supported credential" in inspect("value = " + repr(token))


def test_escaped_literal_and_private_key_are_blocked():
    token = "hf_" + "z" * 34
    escaped = "".join("\\x" + format(ord(c), "02x") for c in token)
    assert "possible supported credential" in inspect('value = "' + escaped + '"')
    header = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    assert "possible supported credential" in inspect("value = " + repr(header))


def test_url_credentials_are_blocked_but_runtime_configuration_is_allowed():
    credential = "postgresql://user:" + "invented-password" + "@example.invalid/db"
    assert "possible supported credential" in inspect("url = " + repr(credential))
    assert not inspect('import os\nurl = os.environ["DATABASE_URL"]\n')
    assert not inspect('url = f"postgresql://{user}:{password}@{host}/db"\n')


@pytest.mark.parametrize("octets", [
    (10, 24, 7, 9), (172, 16, 2, 8), (172, 31, 9, 1), (192, 168, 5, 6),
    (100, 64, 1, 1), (100, 127, 200, 1), (169, 254, 1, 8),
])
def test_private_addresses_are_blocked(octets):
    address = ".".join(map(str, octets))
    assert "private network address or server name" in inspect("host = " + repr(address))


@pytest.mark.parametrize("value", ["127.0.0.1", "192.0.2.7", "198.51.100.9", "203.0.113.4", "example.invalid"])
def test_loopback_and_documentation_addresses_are_allowed(value):
    assert not inspect("host = " + repr(value))


def test_private_ipv6_server_and_home_paths_are_blocked():
    for address in ("fd" + "23::9", "fe" + "80::2", "cs-" + "node-" + "88b", "compute." + "internal"):
        assert "private network address or server name" in inspect("host = " + repr(address))
    for path in ("/" + "home" + "/synthetic_account/work", "/" + "Users" + "/synthetic_account",
                 "C:" + "\\Users\\synthetic_account\\work", "/var/" + "folders/fictional/temp"):
        assert "private home or device path" in inspect("path = " + repr(path))


def test_personal_number_pattern_is_blocked():
    invented = "800" + "101-" + "1234567"
    assert "possible personal identification number" in inspect("value = " + repr(invented))


def record():
    label = "synthetic_record"
    return dict(subject_id=label, phq9=7, gad7=4)


@pytest.mark.parametrize("representation", ["dict", "dict_call", "rows", "columns", "tuples", "json", "json_lines", "csv", "tsv", "pipe"])
def test_record_exports_disguised_as_python_are_blocked(representation):
    item = record()
    if representation == "dict":
        source = "data = " + repr(item)
    elif representation == "dict_call":
        source = "data = dict(" + ", ".join(key+"="+repr(value) for key, value in item.items()) + ")"
    elif representation == "rows":
        source = "data = " + repr([item, item])
    elif representation == "columns":
        source = "data = " + repr({key: [value] for key, value in item.items()})
    elif representation == "tuples":
        source = "data = " + repr([tuple(item), tuple(item.values())])
    elif representation == "json":
        source = "blob = " + repr(json.dumps([item]))
    elif representation == "json_lines":
        source = "blob = " + repr(json.dumps(item) + "\n" + json.dumps(item))
    else:
        sep = {"csv": ",", "tsv": "\t", "pipe": "|"}[representation]
        table = sep.join(item) + "\n" + sep.join(map(str, item.values())) + "\n"
        source = "blob = " + repr(table)
    assert any("record or result table" in reason for reason in inspect(source))


def test_literal_model_result_table_is_blocked():
    model_name = "synthetic_model"
    values = dict(model=model_name, auroc=.75, brier=.2)
    assert any("record or result table" in reason for reason in inspect("result = " + repr(values)))


def test_schemas_formulas_and_generated_synthetic_fixtures_are_allowed():
    schema = 'schema = {"subject_id": "string", "phq9": "float64"}\n'
    assert not inspect(schema)
    fixture = ('import numpy as np\n'
               'rng = np.random.default_rng(4)\n'
               'rows = [dict(subject_id=f"synthetic_{i}", phq9=int(rng.integers(0, 28))) '
               'for i in range(8)]\n')
    assert not inspect(fixture)
    assert not inspect('def risk(z):\n    return 1 / (1 + np.exp(-z))\n')


def test_syntax_is_checked_without_echoing_source():
    assert inspect('broken = {') == ["invalid Python source"]


def git(folder, *args):
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()


def setup_repo(folder):
    git(folder, "init", "-q")
    git(folder, "config", "user.name", "Synthetic Test")
    git(folder, "config", "user.email", "synthetic@example.invalid")
    (folder / "src").mkdir()


def test_staged_bytes_checked_and_matching_value_not_echoed(tmp_path, monkeypatch, capsys):
    setup_repo(tmp_path)
    secret = "hf_" + "z" * 34
    source = tmp_path / "src" / "example.py"
    source.write_text("value = " + repr(secret))
    git(tmp_path, "add", "src/example.py")
    source.write_text('value = "safe"\n')
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    assert checker.main(["--staged"]) == 1
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "possible supported credential" in output.err


def test_intermediate_record_export_is_rejected_after_clean_tip(tmp_path, monkeypatch, capsys):
    setup_repo(tmp_path)
    source = tmp_path / "src" / "example.py"
    source.write_text('value = "safe"\n')
    git(tmp_path, "add", "src/example.py")
    git(tmp_path, "commit", "-qm", "Initial synthetic code")
    base = git(tmp_path, "rev-parse", "HEAD")
    source.write_text("rows = " + repr([record()]))
    git(tmp_path, "add", "src/example.py")
    git(tmp_path, "commit", "-qm", "Synthetic forbidden intermediate")
    source.write_text('value = "safe again"\n')
    git(tmp_path, "add", "src/example.py")
    git(tmp_path, "commit", "-qm", "Synthetic clean tip")
    tip = git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"refs/heads/main {tip} refs/heads/main {base}\n"))
    assert checker.main(["--push"]) == 1
    output = capsys.readouterr()
    assert "embedded literal record" in output.err
    assert "synthetic_record" not in output.out + output.err
