from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT.parent
TARGET_DIR = ROOT / "models"
TARGET_DIR.mkdir(exist_ok=True)

for file_name in ["cross_encoder_scorer.pt", "bi_encoder_head.pt"]:
    source_path = SOURCE_DIR / file_name
    target_path = TARGET_DIR / file_name
    if source_path.exists():
        shutil.copy2(source_path, target_path)
        print(f"copied: {source_path} -> {target_path}")
    else:
        print(f"missing: {source_path}")
