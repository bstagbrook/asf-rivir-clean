"""
Run Phase 1 test demos (ASF2 and ASF Core 2).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path("/Volumes/StagbrookField/stagbrook_field")


def run(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode


def main():
    failures = 0
    failures += run([sys.executable, str(ROOT / "asf_core4" / "asf2_demo.py")]) != 0
    failures += run([sys.executable, str(ROOT / ".asf_core2.py")]) != 0
    if failures:
        print(f"FAIL: {failures} test(s) failed")
        sys.exit(1)
    print("PASS: Phase 1 demos")


if __name__ == "__main__":
    main()
