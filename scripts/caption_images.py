import argparse, json, os
from tqdm import tqdm
from src.utils.data import CocoValDataset
from src.model.blip_captioner import BLIPCaptioner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--model_name", default="Salesforce/blip-image-captioning-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    ds = CocoValDataset(args.data_root, args.split, args.max_images)
    cap = BLIPCaptioner(args.model_name, args.device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for item in tqdm(ds, total=len(ds)):
            hyp = cap.caption(item["image"])
            rec = {"image_id": item["image_id"], "file_name": item["file_name"], "hypothesis": hyp, "references": item["captions"]}
            f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
