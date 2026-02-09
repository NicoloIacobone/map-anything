import os
import subprocess
from pathlib import Path

LOCAL_ROOT = "/scratch2/nico/distillation/dataset/converted/wai_data/blendedmvs"
REMOTE_ROOT = "/cluster/scratch/niacobone/distillation/dataset/processed/blendedmvs"
REMOTE_USER_HOST = "niacobone@euler.ethz.ch"

def remote_path_exists(remote_path):
    cmd = [
        "ssh", REMOTE_USER_HOST,
        f"test -d '{remote_path}/moge' && echo exists || echo missing"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "exists" in result.stdout

def local_path_exists(local_path):
    return (local_path / "moge").is_dir()

def copy_from_remote(scene):
    remote_scene = f"{REMOTE_USER_HOST}:{REMOTE_ROOT}/{scene}/moge"
    local_scene = f"{LOCAL_ROOT}/{scene}/moge"
    # Copia la cartella moge
    subprocess.run([
        "rsync", "-avz", "--progress", remote_scene, f"{LOCAL_ROOT}/{scene}/"
    ])
    # Copia il file scene_meta.json
    remote_json = f"{REMOTE_USER_HOST}:{REMOTE_ROOT}/{scene}/scene_meta.json"
    local_json = f"{LOCAL_ROOT}/{scene}/scene_meta.json"
    subprocess.run([
        "rsync", "-avz", "--progress", remote_json, local_json
    ])

def main():
    for scene in os.listdir(LOCAL_ROOT):
        local_scene_path = Path(LOCAL_ROOT) / scene
        if not local_scene_path.is_dir():
            continue
        # Controlla se la cartella moge esiste in locale o su server
        if local_path_exists(local_scene_path) or remote_path_exists(f"{REMOTE_ROOT}/{scene}"):
            print(f"SKIP: {scene} (moge già presente)")
            continue
        print(f"COPIO: {scene}")
        copy_from_remote(scene)

if __name__ == "__main__":
    main()