"""Download ProsQA data (facebookresearch/coconut) and the four best checkpoints
(bmarti44/coconut-curriculum-checkpoints) into data/ and models/."""

import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lrspec.paths import DATA, MODELS  # noqa: E402

PROSQA = "https://raw.githubusercontent.com/facebookresearch/coconut/main/data"
HF_REPO = "bmarti44/coconut-curriculum-checkpoints"
BEST = {"cot-baseline": "M1", "coconut": "M2",
        "pause-curriculum": "M3", "pause-multipass": "M4"}


def main():
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    for split in ["train", "valid", "test"]:
        dst = DATA / f"prosqa_{split}.json"
        if not dst.exists():
            print(f"downloading prosqa_{split}.json ...")
            urllib.request.urlretrieve(f"{PROSQA}/prosqa_{split}.json", dst)
    from huggingface_hub import hf_hub_download

    for name in BEST:
        dst = MODELS / f"{name}_best.pt"
        if not dst.exists():
            print(f"downloading {name}/checkpoint_best ...")
            p = hf_hub_download(HF_REPO, f"{name}/checkpoint_best")
            shutil.copy(p, dst)
    print("done.")


if __name__ == "__main__":
    main()
