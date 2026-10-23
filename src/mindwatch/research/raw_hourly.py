"""Explicit local wrapper for the original wide-CSV hourly stream parser."""
from pathlib import Path
import json
import os
import shutil
import subprocess
from mindwatch.pipeline import private_output
from .privacy import secure_dir


def aggregate_csv(input_path,output_path,n_columns,blood_pressure=False,compiler=None,timeout=1800):
    """Aggregate an explicitly local-time CSV; publish only a complete valid output.

    The first header must name time/date/timestamp/regdt (or the documented '-'
    placeholder). Exactly n_columns unique channel headers must follow. Explicit
    offsets require caller-side resolution; they are never guessed or stripped.
    """
    input_path=Path(input_path).resolve();output_path=private_output(output_path)
    if input_path==output_path or output_path.exists():raise ValueError("Use a new derived output path")
    if not input_path.is_file():raise ValueError("Existing local CSV input required")
    if isinstance(n_columns,bool) or not isinstance(n_columns,int) or not 1<=n_columns<=100000:raise ValueError("Channel-column count must be 1 through 100000")
    if not isinstance(blood_pressure,bool):raise ValueError("Explicit boolean pressure-pair mode required")
    if not isinstance(timeout,(int,float)) or isinstance(timeout,bool) or not 0<timeout<=86400:raise ValueError("Positive bounded parser timeout required")
    compiler=compiler or shutil.which("clang++") or shutil.which("g++")
    if not compiler:raise RuntimeError("A C++17 compiler is required")
    secure_dir(output_path.parent)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hourly-parser-",dir=output_path.parent) as scratch:
        binary=Path(scratch)/"hourly_stream"
        try:
            compiled=subprocess.run([compiler,"-std=c++17","-O2",str(Path(__file__).with_name("hourly_stream.cpp")),"-o",str(binary)],capture_output=True,timeout=120)
        except (OSError,subprocess.SubprocessError):
            raise RuntimeError('Hourly parser compiler could not finish') from None
        if compiled.returncode:raise RuntimeError("Hourly parser compilation failed")
        pending=Path(scratch)/"pending.csv"
        fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
        try:
            with input_path.open("rb") as stream:
                before=os.fstat(stream.fileno())
                processed=subprocess.run([str(binary),str(n_columns),str(int(blood_pressure)),str(pending)],stdin=stream,capture_output=True,timeout=timeout)
                after=os.fstat(stream.fileno());current=input_path.stat()
        except (OSError,subprocess.SubprocessError):
            raise RuntimeError('Hourly parser input or process could not finish') from None
        signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if signature(before)!=signature(after) or signature(before)!=signature(current):
            raise ValueError('Hourly input changed during parsing')
        if processed.returncode:
            raise ValueError("Hourly parsing failed; check local timestamps, CSV schema, numeric fields and I/O")
        audit=json.loads(processed.stderr)
        if not pending.is_file() or pending.stat().st_size==0:raise RuntimeError("Hourly parser produced no complete output")
        # Same filesystem and exclusive publication; failures never leave a valid-looking output.
        os.link(pending,output_path)
        return audit
