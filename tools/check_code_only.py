"""Fail closed on tracked/staged non-code material before public publication."""
import argparse
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
PRIVATE_PATH = re.compile(r"/(?:Users|home)/[^\s/]+/|[A-Za-z]:\\Users\\")
CREDENTIAL = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|sk-[A-Za-z0-9_-]{32,})")


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
    try:
        value = content.decode("utf-8")
    except UnicodeDecodeError:
        return [*reasons, "not UTF-8 text"]
    if PRIVATE_PATH.search(value):
        reasons.append("private home path")
    if CREDENTIAL.search(value):
        reasons.append("possible credential")
    return reasons


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def inspect_commit(commit):
    failures = []
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect index bytes, not working-tree bytes")
    parser.add_argument("--push", action="store_true", help="Inspect every new commit, including intermediate commits")
    args = parser.parse_args(argv)
    failures = []
    if args.push:
        for update in sys.stdin:
            _, local_sha, _, _ = update.split()
            if local_sha == "0"*40:
                continue
            if not re.fullmatch(r"[0-9a-f]{40}", local_sha):
                raise ValueError("Invalid push ref")
            commits = git("rev-list", local_sha, "--not", "--remotes").decode().splitlines()
            for commit in commits:
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
