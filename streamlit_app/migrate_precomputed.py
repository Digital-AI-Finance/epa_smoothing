"""
Migration Script: Move flat precomputed files to n_2000/ subdirectory.

This one-time script migrates the existing flat file structure:
    precomputed_data/
        metadata.pkl.gz
        sigma_0.10.pkl.gz
        ...

To the new n_points-aware structure:
    precomputed_data/
        n_2000/
            metadata.pkl.gz
            sigma_0.10.pkl.gz
            ...

Usage:
    python migrate_precomputed.py
"""

import shutil
from pathlib import Path


def migrate():
    """Move existing flat files to n_2000/ subdirectory."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'precomputed_data'

    if not data_dir.exists():
        print("No precomputed_data directory found. Nothing to migrate.")
        return

    # Check if already migrated (n_2000 subdir exists)
    n_2000_dir = data_dir / 'n_2000'
    if n_2000_dir.exists():
        print(f"Already migrated: {n_2000_dir} exists")
        return

    # Find files to migrate (*.pkl.gz in root of precomputed_data)
    files_to_migrate = list(data_dir.glob('*.pkl.gz'))

    if not files_to_migrate:
        print("No .pkl.gz files found in precomputed_data/. Nothing to migrate.")
        return

    print(f"Found {len(files_to_migrate)} files to migrate:")
    for f in files_to_migrate:
        print(f"  - {f.name}")

    # Create n_2000 subdirectory
    n_2000_dir.mkdir(exist_ok=True)
    print(f"\nCreated: {n_2000_dir}")

    # Move files
    for f in files_to_migrate:
        dest = n_2000_dir / f.name
        shutil.move(str(f), str(dest))
        print(f"  Moved: {f.name} -> n_2000/{f.name}")

    print(f"\nMigration complete! {len(files_to_migrate)} files moved to n_2000/")


if __name__ == '__main__':
    migrate()
