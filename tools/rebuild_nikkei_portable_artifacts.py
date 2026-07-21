from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.nikkei_artifact import load_portable_artifact
from services.nikkei_dual_model import reevaluate_dual_market_model


def main() -> None:
    result = reevaluate_dual_market_model()
    packages, manifest = load_portable_artifact()
    if sorted(packages["contexts"]) != ["after_close", "intraday"]:
        raise RuntimeError("both Nikkei prediction contexts were not saved")
    command = [
        sys.executable,
        "-c",
        "from services.nikkei_artifact import load_portable_artifact; "
        "p,m=load_portable_artifact(); print(sorted(p['contexts']),m['model_sha256'])",
    ]
    child = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    summary = {
        "reevaluation": result.get("reevaluation"),
        "manifest": manifest,
        "new_process_load": child.stdout.strip(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()