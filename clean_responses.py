import csv
import json
from pathlib import Path
from typing import Iterable, Dict, Any

INPUT_PATH = Path("data/bronze/responses.jsonl")
OUTPUT_PATH = Path("data/silver/responses.csv")


def load_responses(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_segment(text: str) -> str:
    if not text:
        return ""
    idx = text.find("(")
    if idx == -1:
        return ""
    return text[idx + 1 :]


def main() -> None:
    entries = []
    for record in load_responses(INPUT_PATH):
        clean_response = extract_segment(record.get("response", ""))
        word_count = len(clean_response.split()) if clean_response else 0
        entries.append(
            {
                "persona_key": record.get("persona_key"),
                "iteration": record.get("iteration"),
                "response": clean_response,
                "word_count": word_count,
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["persona_key", "iteration", "response", "word_count"],
        )
        writer.writeheader()
        writer.writerows(entries)

    print(f"Wrote {len(entries)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
