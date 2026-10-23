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
