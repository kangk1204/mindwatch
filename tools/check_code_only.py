"""Check code-only paths and recognizable private content before publication.

This is a bounded publication guard, not a complete data-loss prevention system.
It recognizes supported credentials, private endpoints and literal record/table
exports. Arbitrarily encoded data, unfamiliar identifiers and small numeric
arrays still need human diff review. No matching private content is printed.
"""
import ast
import argparse
import csv
import io
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitignore", "LICENSE", "README.md", "config.yaml", "requirements.txt", "pyproject.toml",
    "generate_model_diagram_mpl.py", "generate_preprocessing_diagram.py",
    "visualize_data_coverage.py", "visualize_interactive.py", "visualize_model_results.py",
    "visualize_phq9_analysis.py",
}
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[^\s/]+(?:/|\b)|/var/folde[r]s/|[A-Za-z]:\\Users\\", re.I)
CREDENTIAL = re.compile(
    r"(?:\bgh[pousr]_[A-Za-z0-9]{24,}|\bgithub_pat_[A-Za-z0-9_]{30,}"
    r"|\bhf_[A-Za-z0-9]{30,}|\b(?:AKIA|ASIA)[A-Z0-9]{16}"
    r"|-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"
    r"|\bsk-[A-Za-z0-9_-]{32,}|\bxox[baprs]-[A-Za-z0-9-]{20,}"
    r"|\bAIza[A-Za-z0-9_-]{35}|\b(?:sk|rk)_live_[A-Za-z0-9]{24,}"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{32,})", re.I)
URL_CREDENTIAL = re.compile(
    r"\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://"
    r"[^/\s:'\"<>{}]+:[^@\s'\"<>{}]+@", re.I)
PRIVATE_HOST = re.compile(r"\bcs[-_]node[-_]\d+[a-z]?\b|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|local|lan)\b", re.I)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:])(?:f[cd][0-9a-f]{2}|fe80):[0-9a-f:]+(?![\w:])", re.I)
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    (0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0x64400000, 10),
    (0xA9FE0000, 16), (0xFC << 120, 7), (0xFE80 << 112, 10)))
PERSONAL_NUMBER = re.compile(r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[- ]?[1-8]\d{6}(?!\d)")
IDENTITY_FIELDS = {"subject_id", "participant_id", "patient_id", "person_id", "record_id",
                   "full_name", "patient_name", "participant_name"}
MEASUREMENT_FIELDS = {"phq9", "phq_9", "phq_total", "gad7", "gad_7", "isi", "total",
                      "target", "prediction", "predicted_phq9", "ema_depression",
                      "ema_anxiety", "ema_sleep", "ema_stress", "steps", "heart_rate"}
METRIC_FIELDS = {"auroc", "auc", "auprc", "brier", "mae", "rmse", "sensitivity", "specificity"}


def _private_endpoint(value):
    if PRIVATE_HOST.search(value):
        return True
    for pattern in (IPV4, IPV6):
        for candidate in pattern.findall(value):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if any(address.version == net.version and address in net for net in PRIVATE_NETWORKS):
                return True
    return False


def _number(value):
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _numbers(value):
    if isinstance(value, (list, tuple)):
        return any(_numbers(item) for item in value)
    return _number(value)


def _literal_records(value):
    """Recognize complete literal row/column exports, not runtime schemas."""
    if isinstance(value, dict):
        fields = {str(k).strip().lower(): v for k, v in value.items()}
        if fields.keys() & IDENTITY_FIELDS and any(_numbers(fields[k]) for k in fields.keys() & MEASUREMENT_FIELDS):
            return True
        # Stored model score tables have no patient ID but are still results.
        if fields.keys() & {"model", "candidate", "model_name"} and sum(
                _numbers(fields[k]) for k in fields.keys() & METRIC_FIELDS) >= 2:
            return True
        return any(_literal_records(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        if len(value) > 1 and isinstance(value[0], (list, tuple)):
            header = value[0]
            if all(isinstance(k, str) for k in header):
                for row in value[1:]:
                    if isinstance(row, (list, tuple)) and len(row) == len(header):
                        if _literal_records(dict(zip(header, row))):
                            return True
        return any(_literal_records(row) for row in value)
    return False


def _serialized_records(value):
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            if _literal_records(json.loads(stripped)):
                return True
        except (ValueError, RecursionError):
            pass
        if "\n" in stripped:
            for line in stripped.splitlines():
                try:
                    if _literal_records(json.loads(line)):
                        return True
                except (ValueError, RecursionError):
                    continue
    if "\n" in value:
        for delimiter in (",", "\t", "|"):
            try:
                rows = list(csv.reader(io.StringIO(value), delimiter=delimiter))
            except csv.Error:
                continue
            if _literal_records(rows):
                return True
    return False


def _text_reasons(value):
    reasons = []
    if PRIVATE_PATH.search(value):
        reasons.append("private home or device path")
    if CREDENTIAL.search(value) or URL_CREDENTIAL.search(value):
        reasons.append("possible supported credential")
    if _private_endpoint(value):
        reasons.append("private network address or server name")
    if PERSONAL_NUMBER.search(value):
        reasons.append("possible personal identification number")
    if _serialized_records(value):
        reasons.append("embedded record or result table")
    return reasons


def _python_reasons(value):
    try:
        tree = ast.parse(value)
    except (SyntaxError, ValueError, RecursionError):
        return ["invalid Python source"]
    reasons = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Also inspect decoded literals (escaped newlines or Unicode).
            reasons.extend(_text_reasons(node.value))
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple)):
            try:
                literal = ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            if _literal_records(literal):
                reasons.append("embedded literal record or result table")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            if node.args or any(k.arg is None for k in node.keywords):
                continue
            try:
                literal = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            except (ValueError, TypeError, SyntaxError, RecursionError):
                continue
            if _literal_records(literal):
                reasons.append("embedded literal record or result table")
    return reasons


def allowed(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        return False
    if path in ROOT_FILES:
        return True
    if path in (".githooks/pre-commit", ".githooks/pre-push", ".github/workflows/code-only.yml"):
        return True
    return bool(p.parts and p.parts[0] in ("src", "tests", "tools") and
                (p.suffix == ".py" or path == "src/mindwatch/research/hourly_stream.cpp"))


def inspect_content(path, content, mode="100644"):
    reasons = []
    if not allowed(path):
        reasons.append("path is outside the code-only allowlist")
    if mode not in ("100644", "100755"):
        reasons.append("symlinks/submodules are forbidden")
    if len(content) > 300_000 or b"\x00" in content:
        reasons.append("binary or oversized file")
        return reasons
    try:
        value = content.decode("utf-8")
    except UnicodeDecodeError:
        return [*reasons, "not UTF-8 text"]
    reasons.extend(_text_reasons(value))
    if PurePosixPath(path).suffix == ".py":
        reasons.extend(_python_reasons(value))
    return list(dict.fromkeys(reasons))


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def inspect_commit(commit):
    reasons = inspect_content("README.md", git("cat-file", "commit", commit))
    failures = [("[commit metadata]", reasons)] if reasons else []
    for record in git("ls-tree", "-r", "-z", commit).split(b"\x00"):
        if not record:
            continue
        info, raw_path = record.split(b"\t", 1)
        mode, kind, oid = info.decode().split()
        path = raw_path.decode("utf-8")
        content = git("cat-file", "blob", oid) if kind == "blob" else b""
        reasons = inspect_content(path, content, mode)
        if reasons:
            failures.append((path, reasons))
    return failures


def inspect_tag_chain(revision):
    failures, seen = [], set()
    while git("cat-file", "-t", revision).strip() == b"tag":
        if revision in seen:
            raise ValueError("Invalid tag chain")
        seen.add(revision)
        content = git("cat-file", "tag", revision)
        reasons = inspect_content("README.md", content)
        if reasons:
            failures.append(("[tag metadata]", reasons))
        first = content.splitlines()[0].decode("ascii")
        if not re.fullmatch(r"object [0-9a-f]{40}", first):
            raise ValueError("Invalid tag target")
        revision = first.split()[1]
    if git("cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("Only commits and tags pointing to commits may be published")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect index bytes, not working-tree bytes")
    parser.add_argument("--push", action="store_true", help="Inspect every new commit, including intermediate commits")
    args = parser.parse_args(argv)
    failures = []
    if args.push:
        inspected = set()
        for update in sys.stdin:
            fields = update.split()
            if len(fields) != 4:
                raise ValueError("Invalid push update")
            _, local_sha, _, remote_sha = fields
            if local_sha == "0"*40:
                continue
            if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (local_sha, remote_sha)):
                raise ValueError("Invalid push ref")
            # A local remote-tracking ref is not evidence that the receiving
            # branch already has safe content (it may name a private remote).
            failures.extend(inspect_tag_chain(local_sha))
            revisions = (local_sha,) if remote_sha == "0"*40 else (local_sha, "^"+remote_sha)
            commits = git("rev-list", *revisions).decode().splitlines()
            # Recheck the outgoing tip even for an already-reachable update.
            commits = [git("rev-parse", local_sha+"^{commit}").decode().strip(), *commits]
            for commit in commits:
                if commit in inspected:
                    continue
                inspected.add(commit)
                failures.extend(inspect_commit(commit))
        if failures:
            for path, reasons in failures:
                print(path+": "+", ".join(reasons), file=sys.stderr)
            return 1
        print("Code-only outgoing-commit check passed.")
        return 0
    for record in git("ls-files", "--stage", "-z").split(b"\x00"):
        if not record:
            continue
        info, raw_path = record.split(b"\t", 1)
        mode, _, stage = info.decode().split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            failures.append((path, ["unmerged entry"]))
            continue
        location = ROOT/path
        if args.staged:
            content = git("show", ":"+path)
        elif not location.exists() and not location.is_symlink():
            continue
        elif location.is_symlink():
            content, mode = b"", "120000"
        else:
            content = location.read_bytes()
        reasons = inspect_content(path, content, mode)
        if reasons:
            failures.append((path, reasons))
    if failures:
        # Never echo matching secret or patient content.
        for path, reasons in failures:
            print(path+": "+", ".join(reasons), file=sys.stderr)
        return 1
    print("Code-only tracked-file check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
