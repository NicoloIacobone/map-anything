#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


def parse_scene_split_file(split_file: Path):
    """
    Parse file format like:
    TRAIN
    ['id1' 'id2' ...]

    TEST
    ['idA' 'idB' ...]
    """
    sections = {"TRAIN": [], "TEST": []}
    current = None
    token_re = re.compile(r"'([^']+)'")

    with split_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("TRAIN"):
                current = "TRAIN"
                continue
            if upper.startswith("TEST"):
                current = "TEST"
                continue

            if current is None:
                continue

            sections[current].extend(token_re.findall(line))

    # Deduplica preservando ordine
    for k in sections:
        sections[k] = list(dict.fromkeys(sections[k]))

    return sections


def check_split(split_name, scene_ids, roots, mode="any", show_missing=20):
    """
    mode='any': scena valida se esiste in almeno uno dei root
    mode='all': scena valida solo se esiste in tutti i root
    """
    missing_global = []
    missing_per_root = {str(r): [] for r in roots}

    for sid in scene_ids:
        exists_flags = []
        for r in roots:
            ok = (r / sid).is_dir()
            exists_flags.append(ok)
            if not ok:
                missing_per_root[str(r)].append(sid)

        if mode == "any":
            ok_global = any(exists_flags)
        else:
            ok_global = all(exists_flags)

        if not ok_global:
            missing_global.append(sid)

    present_global = len(scene_ids) - len(missing_global)

    print(f"\n[{split_name}]")
    print(f"Attese: {len(scene_ids)}")
    print(f"Presenti ({mode}): {present_global}")
    print(f"Mancanti ({mode}): {len(missing_global)}")

    if missing_global:
        print(f"Prime mancanti ({mode}, max {show_missing}): {missing_global[:show_missing]}")

    print("\nDettaglio per root:")
    for root_str, miss_list in missing_per_root.items():
        present = len(scene_ids) - len(miss_list)
        print(f"- {root_str}")
        print(f"  presenti: {present}")
        print(f"  mancanti: {len(miss_list)}")
        if miss_list:
            print(f"  prime mancanti (max {show_missing}): {miss_list[:show_missing]}")

    return len(missing_global)


def main():
    parser = argparse.ArgumentParser(description="Check BlendedMVS scene folders from split file.")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data_processing/blendedmvs_scene_split.txt"),
        help="Path del file con TRAIN/TEST scene IDs",
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[
            Path("/scratch2/nico/distillation/dataset/blendedmvs/blendedmvs"),
            Path("/scratch2/nico/distillation/dataset/converted/wai_data/blendedmvs"),
        ],
        help="Root che contiene le cartelle scene (ripetibile: --root /path1 --root /path2)",
    )
    parser.add_argument(
        "--mode",
        choices=["any", "all"],
        default="any",
        help="any=ok se scena esiste in almeno un root, all=ok solo se esiste in tutti",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="Quante scene mancanti mostrare in output",
    )
    args = parser.parse_args()

    if not args.split_file.is_file():
        print(f"ERRORE: split file non trovato: {args.split_file}")
        sys.exit(2)

    roots = [r for r in args.root]
    for r in roots:
        if not r.exists():
            print(f"ATTENZIONE: root non esiste: {r}")

    sections = parse_scene_split_file(args.split_file)

    n_train = len(sections["TRAIN"])
    n_test = len(sections["TEST"])
    print(f"TRAIN: {n_train} scene")
    print(f"TEST : {n_test} scene")
    print(f"TOTALE: {n_train + n_test} scene")

    missing_train = check_split("TRAIN", sections["TRAIN"], roots, mode=args.mode, show_missing=args.show_missing)
    missing_test = check_split("TEST", sections["TEST"], roots, mode=args.mode, show_missing=args.show_missing)

    total_missing = missing_train + missing_test
    print(f"\nMancanti totali ({args.mode}): {total_missing}")

    # exit code non-zero se manca qualcosa
    sys.exit(1 if total_missing > 0 else 0)


if __name__ == "__main__":
    main()