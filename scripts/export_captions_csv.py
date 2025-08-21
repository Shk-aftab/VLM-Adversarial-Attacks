# scripts/export_captions_csv.py
import argparse, json, pandas as pd

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adversarial", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    base = load_jsonl(args.baseline)
    adv = load_jsonl(args.adversarial)

    bmap = {b["file_name"]: b for b in base}
    rows = []
    for a in adv:
        b = bmap.get(a["file_name"])
        if not b: 
            continue
        rows.append({
            "file_name": a["file_name"],
            "baseline_caption": b.get("hypothesis"),
            "adversarial_caption": a.get("hypothesis_adv"),
            "reference_caption": (b.get("references") or [""])[0],
        })

    df = pd.DataFrame(rows, columns=["file_name","baseline_caption","adversarial_caption","reference_caption"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()


python -m scripts.export_captions_csv --baseline eval/baseline_1000.jsonl --adversarial eval/fgsm_eps4_255_1000.jsonl --out_csv eval/fgsm_eps4_255_1000_captions.csv
