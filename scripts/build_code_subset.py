"""Build data/icd10cm_subset.json from the official CDC/NCHS ICD-10-CM code file.

Download "icd10cm-code-descriptions-<year>.zip" from
https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/<year>/ , unzip it, and run:

    python scripts/build_code_subset.py /path/to/icd10cm-codes-2027.txt

Each line of that file is a code without its dot, padded to 7 characters, then the description.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icd_mock.dataset import ICD_RELEASE, data_dir, load_cases, used_codes  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    official: dict[str, str] = {}
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            official[line[:7].strip()] = line[8:].strip()

    wanted = sorted(used_codes(load_cases()))
    missing = [c for c in wanted if c.replace(".", "") not in official]
    if missing:
        print(f"Not in the official file: {', '.join(missing)}")
        return 1

    subset = {
        "release": ICD_RELEASE,
        "source": "https://ftp.cdc.gov/pub/health_statistics/nchs/publications/ICD10CM/",
        "codes": {code: official[code.replace(".", "")] for code in wanted},
    }
    out = data_dir() / "icd10cm_subset.json"
    out.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(wanted)} codes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
