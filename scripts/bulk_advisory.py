"""Compose advisories for a CSV of recipients; this script never sends anything.

Input CSV columns: ``phone`` (E.164), ``district`` (name or id), ``crop``, ``lang`` (en, hi, ta or te; default en).
Output CSV (default ``data/interim/advisories_<as-of>.csv``): one row per input row with the composed text, its
length, SMS parts and a ``status`` (``ok``, ``skipped_opt_out`` or an error message). Phone numbers are written
masked unless ``--include-phone`` is given, because the file is meant for review, not delivery. To deliver, pass the
reviewed texts to ``advisory.twilio_client`` (which is a dry run unless ``DRY_RUN=false``).

Run: ``python scripts/bulk_advisory.py recipients.csv --as-of 2024-06-15``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path

from advisory import optout, parse
from advisory.generate import compose, load_templates
from advisory.twilio_client import InvalidRecipient, mask_number, normalise_number
from config import get_path

LOG = logging.getLogger("bulk_advisory")
FIELDS = ["row", "phone", "district_id", "crop", "lang", "text", "chars", "sms_parts", "status"]


def resolve_district(value: str) -> str | None:
    """A district id, or the confidently matched district for a typed name (None if unknown or ambiguous)."""
    aliases, labels = parse.district_index()
    if value in labels:
        return value
    match = parse.match_in_text(value, accept=92.0, confirm=70.0)
    return match.district_id if match is not None and match.confident else None


def process_row(
    index: int, row: dict[str, str], as_of: date, include_phone: bool
) -> dict[str, object]:
    """Compose the advisory for one input row; errors are reported in ``status`` and never raised."""
    out: dict[str, object] = {
        "row": index,
        "phone": "",
        "district_id": "",
        "crop": "",
        "lang": "",
        "text": "",
        "chars": 0,
        "sms_parts": 0,
    }
    try:
        number = normalise_number(row.get("phone", ""))
    except InvalidRecipient as exc:
        out["status"] = f"error: {exc}"
        return out
    out["phone"] = number if include_phone else mask_number(number)
    crop = (row.get("crop") or "").strip().lower()
    lang = (row.get("lang") or "en").strip().lower() or "en"
    district_id = resolve_district((row.get("district") or "").strip())
    out.update(district_id=district_id or "", crop=crop, lang=lang)
    if optout.is_opted_out(number):
        out["status"] = "skipped_opt_out"
    elif district_id is None:
        out["status"] = f"error: district {row.get('district')!r} not found or ambiguous"
    elif crop not in load_templates()["crops"]:
        out["status"] = f"error: unknown crop {crop!r}"
    elif lang not in load_templates()["languages"]:
        out["status"] = f"error: unsupported language {lang!r}"
    else:
        try:
            advisory = compose(district_id, crop, as_of, lang)
        except (ValueError, FileNotFoundError) as exc:
            out["status"] = f"error: {exc}"
        else:
            out.update(
                text=advisory.text,
                chars=len(advisory.text),
                sms_parts=advisory.sms_segments,
                status="ok",
            )
    return out


def run(
    source: Path, as_of: date, output: Path | None, include_phone: bool
) -> tuple[Path, dict[str, int]]:
    """Process the whole file; returns the output path and a count per status."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    results = [process_row(i, row, as_of, include_phone) for i, row in enumerate(rows, start=1)]
    target = output or get_path("data_interim") / f"advisories_{as_of.isoformat()}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    counts: dict[str, int] = {}
    for result in results:
        key = "ok" if result["status"] == "ok" else str(result["status"]).split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    return target, counts


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns 1 when any row could not be composed."""
    parser = argparse.ArgumentParser(
        description="Compose advisories for a CSV of recipients (never sends)."
    )
    parser.add_argument("csv", type=Path, help="input CSV with phone,district,crop,lang")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        required=True,
        help="advisory date, YYYY-MM-DD (within the observed record)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-phone",
        action="store_true",
        help="write full phone numbers instead of masked ones",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    target, counts = run(args.csv, args.as_of, args.output, args.include_phone)
    LOG.info("wrote %s: %s", target, ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if any(key not in {"ok", "skipped_opt_out"} for key in counts) else 0


if __name__ == "__main__":
    sys.exit(main())
