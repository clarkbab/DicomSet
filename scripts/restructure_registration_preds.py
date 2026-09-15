"""Restructure registration prediction folders to include series IDs.

Old layout:
    <root>/<fixed_pat>/<fixed_study>/<moving_pat>/<moving_study>/...

New layout:
    <root>/<fixed_pat>/<fixed_study>/<fixed_series>/<moving_pat>/<moving_study>/<moving_series>/...

Usage:
    python restructure_registration_preds.py <root> [--apply] [--series-id series_0]

Without --apply, only the first planned move is printed (dry run).
"""

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import List, Tuple


def plan_moves(root: Path, series_id: str) -> List[Tuple[Path, Path]]:
    moves: List[Tuple[Path, Path]] = []
    for fixed_pat in sorted(p for p in root.iterdir() if p.is_dir()):
        for fixed_study in sorted(p for p in fixed_pat.iterdir() if p.is_dir()):
            for moving_pat in sorted(p for p in fixed_study.iterdir() if p.is_dir()):
                if moving_pat.name == series_id:
                    # Already migrated.
                    continue
                for moving_study in sorted(p for p in moving_pat.iterdir() if p.is_dir()):
                    src = moving_study
                    dst = fixed_study / series_id / moving_pat.name / moving_study.name / series_id
                    moves.append((src, dst))
    return moves


def apply_move(src: Path, dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f'Destination already exists: {dst}')
    # 'dst' is nested inside 'src' when moving_pat == series_id-level, so stage via temp dir.
    tmp = Path(tempfile.mkdtemp(dir=src.parent.parent.parent))  # under fixed_study
    staged = tmp / src.name
    shutil.move(str(src), str(staged))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(dst))
    tmp.rmdir()
    # Clean up empty old moving_pat dir.
    old_moving_pat = src.parent
    if old_moving_pat.exists() and not any(old_moving_pat.iterdir()):
        old_moving_pat.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path, help='Path to .../predictions/registration/patients')
    parser.add_argument('--apply', action='store_true', help='Actually move folders (default: dry run of first move).')
    parser.add_argument('--series-id', default='series_0')
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f'Root does not exist: {root}')

    moves = plan_moves(root, args.series_id)
    print(f'Found {len(moves)} folder(s) to move.')
    if not moves:
        return

    if not args.apply:
        src, dst = moves[0]
        print('DRY RUN - first planned move:')
        print(f'  {src}')
        print(f'  -> {dst}')
        print('Re-run with --apply to perform all moves.')
        return

    for src, dst in moves:
        print(f'{src} -> {dst}')
        apply_move(src, dst)
    print('Done.')


if __name__ == '__main__':
    main()
