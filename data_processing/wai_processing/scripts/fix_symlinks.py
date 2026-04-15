#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Fix symlinks in WAI dataset images directories.

This script updates symlinks that point to old local paths to new cluster paths.
For example: /scratch2/nico/distillation/... → /cluster/scratch/niacobone/distillation/...

Usage:
    python fix_symlinks.py \
        --wai_root /dataset/converted/wai_data/blendedmvs \
        --old_prefix /scratch2/nico/distillation/ \
        --new_prefix /cluster/scratch/niacobone/distillation/
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm


def fix_symlinks_in_dataset(
    wai_root: Path,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Fix symlinks in a WAI dataset.
    
    Args:
        wai_root: Root path of the WAI dataset (e.g., /dataset/converted/wai_data/blendedmvs)
        old_prefix: Old path prefix to replace (e.g., /scratch2/nico/distillation/)
        new_prefix: New path prefix (e.g., /cluster/scratch/niacobone/distillation/)
        dry_run: If True, only report what would be changed without modifying
        verbose: If True, print detailed info about each operation
        
    Returns:
        Dictionary with statistics about the operation
    """
    wai_root = Path(wai_root)
    
    if not wai_root.exists():
        raise FileNotFoundError(f"WAI root not found: {wai_root}")
    
    # Ensure old_prefix ends with /
    old_prefix = old_prefix.rstrip('/') + '/'
    new_prefix = new_prefix.rstrip('/') + '/'
    
    stats = {
        'total_symlinks': 0,
        'fixed_symlinks': 0,
        'skipped_symlinks': 0,  # Already pointing to /cluster/
        'broken_symlinks': 0,   # Target doesn't exist
        'errors': [],
    }
    
    # Find all scene directories
    scene_dirs = sorted([d for d in wai_root.iterdir() if d.is_dir()])
    
    print(f"Scanning {len(scene_dirs)} scenes in {wai_root}")
    
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        images_dir = scene_dir / "images"
        
        if not images_dir.exists():
            continue
        
        # Find all files in images directory
        for image_file in sorted(images_dir.iterdir()):
            if not image_file.is_symlink():
                continue
            
            stats['total_symlinks'] += 1
            
            # Read the current symlink target
            target = os.readlink(image_file)
            
            if verbose:
                print(f"  Found symlink: {image_file.name} -> {target}")
            
            # Check if already points to /cluster/
            if target.startswith('/cluster/'):
                stats['skipped_symlinks'] += 1
                if verbose:
                    print(f"    ✓ Already pointing to /cluster/, skipping")
                continue
            
            # Check if target should be updated
            if old_prefix in target:
                # Replace old prefix with new prefix
                new_target = target.replace(old_prefix, new_prefix)
                
                if verbose:
                    print(f"    Old target: {target}")
                    print(f"    New target: {new_target}")
                
                # Check if new target path would exist (only if not dry_run)
                if not dry_run and not Path(new_target).exists():
                    stats['broken_symlinks'] += 1
                    error_msg = f"New target doesn't exist: {image_file} -> {new_target}"
                    stats['errors'].append(error_msg)
                    print(f"    ✗ WARNING: {error_msg}")
                    continue
                
                # Update the symlink
                if not dry_run:
                    try:
                        image_file.unlink()  # Remove old symlink
                        os.symlink(new_target, image_file)
                        stats['fixed_symlinks'] += 1
                        if verbose:
                            print(f"    ✓ Updated symlink")
                    except Exception as e:
                        error_msg = f"Error updating {image_file}: {e}"
                        stats['errors'].append(error_msg)
                        print(f"    ✗ {error_msg}")
                else:
                    stats['fixed_symlinks'] += 1
                    if verbose:
                        print(f"    [DRY RUN] Would update symlink")
            else:
                stats['skipped_symlinks'] += 1
                if verbose:
                    print(f"    - Old prefix not found in target, skipping")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix symlinks in WAI dataset that point to old local paths"
    )
    parser.add_argument(
        "--wai_root",
        type=str,
        required=True,
        help="Root path of the WAI dataset (e.g., /dataset/converted/wai_data/blendedmvs)"
    )
    parser.add_argument(
        "--old_prefix",
        type=str,
        required=True,
        help="Old path prefix to replace (e.g., /scratch2/nico/distillation/)"
    )
    parser.add_argument(
        "--new_prefix",
        type=str,
        required=True,
        help="New path prefix (e.g., /cluster/scratch/niacobone/distillation/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be changed without modifying files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info about each symlink operation"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WAI Dataset Symlink Fixer")
    print("=" * 80)
    print(f"WAI Root: {args.wai_root}")
    print(f"Old Prefix: {args.old_prefix}")
    print(f"New Prefix: {args.new_prefix}")
    print(f"Dry Run: {args.dry_run}")
    print("=" * 80)
    
    try:
        stats = fix_symlinks_in_dataset(
            wai_root=args.wai_root,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Total symlinks found:  {stats['total_symlinks']}")
    print(f"Fixed symlinks:        {stats['fixed_symlinks']}")
    print(f"Skipped symlinks:      {stats['skipped_symlinks']}")
    print(f"Broken targets:        {stats['broken_symlinks']}")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  - {error}")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No changes were made")
    else:
        print(f"\n✓ Fixed {stats['fixed_symlinks']} symlinks")
    
    return 0 if not stats['errors'] else 1


if __name__ == "__main__":
    sys.exit(main())