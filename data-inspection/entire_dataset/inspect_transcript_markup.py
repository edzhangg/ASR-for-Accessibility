
from pathlib import Path
from collections import Counter, defaultdict
import json
import random
import re

ROOT = Path(r"C:\SAP downsampled database")
random.seed(42)

patterns = {
    "square": re.compile(r"\[[^\]]*\]"),
    "parentheses": re.compile(r"\([^)]*\)"),
    "curly": re.compile(r"\{[^}]*\}"),
}

markup_counts = defaultdict(Counter)
examples = defaultdict(list)

for split in ("Train", "Dev"):
    split_dir = ROOT / split

    # Only original dataset shards
    shards = [
        p for p in split_dir.iterdir()
        if p.is_dir()
        and re.fullmatch(
            r"SpeechAccessibility_\d{4}-\d{2}-\d{2}_\d{3}",
            p.name,
        )
    ]

    for shard in shards:
        for json_path in shard.rglob("*.json"):
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)

            for item in meta.get("Files", []):
                prompt = item.get("Prompt") or {}
                transcript = prompt.get("Transcript") or ""

                for name, pattern in patterns.items():
                    matches = pattern.findall(transcript)

                    if not matches:
                        continue

                    markup_counts[name].update(matches)

                    examples[name].append({
                        "split": split,
                        "etiology": meta.get("Etiology"),
                        "category": prompt.get(
                            "Category Description"
                        ),
                        "transcript": transcript,
                        "filename": item.get("Filename"),
                        "json_path": str(json_path),
                    })

for name in patterns:
    print(f"\n========== {name.upper()} ==========")

    print("\nMost common markup values:")
    for value, count in markup_counts[name].most_common(20):
        print(f"  {count:>6}  {value}")

    print("\nRandom examples:")
    sample = random.sample(
        examples[name],
        min(20, len(examples[name])),
    )

    for record in sample:
        print("\nEtiology:", record["etiology"])
        print("Category:", record["category"])
        print("Transcript:", record["transcript"])
        print("WAV:", record["filename"])
