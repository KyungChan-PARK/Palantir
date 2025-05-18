"""gen_repo_csv.py
Usage:
    poetry run python scripts/gen_repo_csv.py data/source/pocketflow_tck output/pftck_meta.csv
Generates a simple CSV (path,size,ext) listing all files under repo.
"""
import sys
import csv
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/gen_repo_csv.py <repo_dir> <output_csv>")
        sys.exit(1)
    repo = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    rows = []
    for p in repo.rglob('*'):
        if p.is_file():
            rows.append({'path': str(p.relative_to(repo)), 'size': p.stat().st_size, 'ext': p.suffix.lower()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'size', 'ext'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV generated: {out_csv} ({len(rows)} rows)")

if __name__ == '__main__':
    main() 