from huggingface_hub import snapshot_download
import logging

# Abilita logging verboso
logging.basicConfig(level=logging.INFO)

print("[DEBUG] Starting snapshot_download...")

snapshot_download(
    repo_id="facebook/map-anything",
    repo_type="dataset",
    local_dir="/scratch2/nico/distillation/dataset/hf_meta_blendedmvs",
    allow_patterns=[
        "wai_data/blendedmvs/**",
    ],
    max_workers=8,
)

print("[DEBUG] Download completed!")