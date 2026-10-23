"""Explicit local wrapper for the original wide-CSV hourly stream parser."""
from pathlib import Path
import json
import shutil
import subprocess
from mindwatch.pipeline import private_output
from .privacy import secure_dir


def aggregate_csv(input_path,output_path,n_columns,blood_pressure=False,compiler=None):
    input_path=Path(input_path).resolve();output_path=private_output(output_path)
    if input_path==output_path or output_path.exists():raise ValueError("Use a new derived output path")
    if not isinstance(n_columns,int) or n_columns<1:raise ValueError("Positive channel-column count required")
    compiler=compiler or shutil.which("clang++") or shutil.which("g++")
    if not compiler:raise RuntimeError("A C++17 compiler is required")
    secure_dir(output_path.parent)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hourly-parser-",dir=output_path.parent) as scratch:
        binary=Path(scratch)/"hourly_stream"
        compiled=subprocess.run([compiler,"-std=c++17","-O2",str(Path(__file__).with_name("hourly_stream.cpp")),"-o",str(binary)],capture_output=True)
        if compiled.returncode:raise RuntimeError("Hourly parser compilation failed")
        with input_path.open("rb") as stream:
            processed=subprocess.run([str(binary),str(n_columns),str(int(blood_pressure)),str(output_path)],stdin=stream,capture_output=True)
        if output_path.exists():output_path.chmod(0o600)
        if processed.returncode:
            # Keep partial output privately for diagnosis; never return it as valid.
            raise ValueError("Hourly parsing failed; check time ordering, duplicates and calendar syntax")
        return json.loads(processed.stderr)
