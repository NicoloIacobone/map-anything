# import os
# import sys
# from huggingface_hub import snapshot_download
# import numpy as np
# from pathlib import Path

# OUTPUT_DIR = "/scratch2/nico/map-anything-metadata"

# print("[START] Downloading MapAnything metadata from HuggingFace...")
# print(f"[INFO] Target directory: {OUTPUT_DIR}")
# print(f"[INFO] Size: ~115 GB (this will take a while...)")
# print()

# try:
#     # 1. Scarica i metadata con output verboso
#     local_dir = snapshot_download(
#         repo_id="facebook/map-anything",
#         repo_type="dataset",
#         local_dir=OUTPUT_DIR,
#         max_workers=4,  # Ridotto per stabilità
#         force_download=False,  # Non riscarica se già presente
#     )
    
#     print(f"[SUCCESS] Download completed to: {local_dir}")
#     print()
    
#     # 2. Leggi le liste di scene per il training
#     datasets = [
#         "ase", "blendedmvs", "dl3dv", "dynamicreplica", "eth3d",
#         "megadepth", "mpsd", "mvs_synth", "paralleldomain4d",
#         "sailvos3d", "scannetppv2", "spring", "tav2_wb", "unrealstereo4k"
#     ]
    
#     print("[INFO] Checking scene lists...")
#     print()
    
#     total_scenes = 0
#     for dataset in datasets:
#         scene_list_path = os.path.join(OUTPUT_DIR, "train", f"{dataset}_scene_list_train.npy")
        
#         if not os.path.exists(scene_list_path):
#             print(f"[WARN] {dataset}: Scene list not found at {scene_list_path}")
#             continue
        
#         try:
#             train_scenes = np.load(scene_list_path, allow_pickle=True)
#             num_scenes = len(train_scenes)
#             total_scenes += num_scenes
#             print(f"[OK] {dataset:20s}: {num_scenes:5d} training scenes")
#         except Exception as e:
#             print(f"[ERROR] {dataset}: Failed to load - {e}")
    
#     print()
#     print(f"[SUMMARY] Total training scenes across all datasets: {total_scenes}")

# except Exception as e:
#     print(f"[ERROR] Download failed: {e}", file=sys.stderr)
#     print(f"[DEBUG] Current directory: {os.getcwd()}")
#     print(f"[DEBUG] Directory exists: {os.path.exists(OUTPUT_DIR)}")
#     sys.exit(1)


import os
import sys
from huggingface_hub import list_repo_files, hf_hub_url
from pathlib import Path

print("[DEBUG] Testing HuggingFace connection...")
print()

try:
    # Test 1: Verifica connessione a HF
    print("[TEST 1] Listing repo files...")
    files = list_repo_files(
        repo_id="facebook/map-anything",
        repo_type="dataset",
    )
    print(f"[OK] Found {len(files)} files in repo")
    print(f"[SAMPLE] First 5 files: {files[:5]}")
    print()
    
except Exception as e:
    print(f"[ERROR] Cannot access HuggingFace repo: {e}")
    print()
    print("[SOLUTION] Possible issues:")
    print("  1. No internet connection")
    print("  2. HuggingFace is down")
    print("  3. The repo is private and you need to login")
    print()
    print("Try this:")
    print("  huggingface-cli login")
    sys.exit(1)

print("[TEST 2] Checking available scene list files...")
scene_files = [f for f in files if "scene_list_train" in f]
print(f"[OK] Found {len(scene_files)} scene list files")
for f in sorted(scene_files)[:5]:
    print(f"  - {f}")
print()

print("[TEST 3] Downloading single small file to test...")
try:
    from huggingface_hub import hf_hub_download
    
    # Scarica il primo file piccolo per testare
    test_file = scene_files[0] if scene_files else "README.md"
    print(f"[DOWNLOAD] {test_file}...")
    
    path = hf_hub_download(
        repo_id="facebook/map-anything",
        filename=test_file,
        repo_type="dataset",
        local_dir="/tmp/hf_test",
    )
    print(f"[SUCCESS] Downloaded to: {path}")
    
    import os
    size_mb = os.path.getsize(path) / (1024*1024)
    print(f"[INFO] File size: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    sys.exit(1)

# import numpy as np

# arr = np.load('/home/niacobone/Downloads/eth3d_scene_list_test.npy', allow_pickle=True)
# print(type(arr[0]))

# import torch, json
# from mapanything.models.mapanything.model import MapAnything

# Carica il JSON di esempio come base
# cfg = json.load(open('/scratch2/nico/examples/meeting_11_09/class_attributes.json'))

# # Istanzia dal blocco di init (puoi modificare encoders per evitare rete)
# init_args = cfg['class_init_args']
# # Esempio: prova a disattivare torch hub se serve offline (opzionale)
# # init_args['encoder_config']['uses_torch_hub'] = False

# model = MapAnything(**init_args)
# model.eval()

# # Mini input sintetico (1 view, B=1, H=W=128)
# B, H, W = 1, 128, 128
# img = torch.randn(B, 3, H, W)
# views = [{
#   'img': img,
#   'data_norm_type': [model.encoder_config['data_norm_type']],  # tipicamente 'dinov2'
# }]

# with torch.inference_mode():
#     preds = model.infer(
#         views,
#         memory_efficient_inference=True,
#         use_amp=False  # più semplice per il test
#     )

# print('Infer OK. Num views:', len(preds))
# print('Keys view0:', preds[0].keys())
# # Se vuoi verificare che la tua head sia stata chiamata:
# has_attr = hasattr(model, '_last_inst_embeddings')
# print('instance embeddings present:', has_attr)
# if has_attr:
#     print('emb shape:', model._last_inst_embeddings.shape)