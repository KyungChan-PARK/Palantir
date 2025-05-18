"""generate_ge_suite.py
자동 GE Expectation Suite 생성기

사용 예:
    poetry run python scripts/generate_ge_suite.py data/interim/clean.csv

CSV를 스캔하여:
1) 모든 컬럼 null 미허용 여부 결정(비율 기준)
2) 정수/실수/날짜 등 데이터 타입 추론 → 기대 타입 추가
3) 결과 Suite JSON 을 data/quality/expectations/{filename}.json 로 저장
"""
import sys
from pathlib import Path
import pandas as pd
import great_expectations as ge
import json

THRESH_NULL = 0.05  # 5% 이하 null 허용
OUTPUT_DIR = Path("data/quality/expectations")


def generate_suite(csv_path: Path):
    df = pd.read_csv(csv_path)
    gdf = ge.from_pandas(df)

    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio < THRESH_NULL:
            gdf.expect_column_values_to_not_be_null(col)

        # 타입 추론
        if pd.api.types.is_integer_dtype(df[col]):
            gdf.expect_column_values_to_be_of_type(col, "int64")
        elif pd.api.types.is_float_dtype(df[col]):
            gdf.expect_column_values_to_be_of_type(col, "float64")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            gdf.expect_column_values_to_match_strftime_format(col, "%Y-%m-%d %H:%M:%S", mostly=0.9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{csv_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(gdf.to_json_dict(), f, ensure_ascii=False, indent=2)
    print(f"✅ Expectation suite saved → {out_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_ge_suite.py <csv_path>")
        sys.exit(1)
    generate_suite(Path(sys.argv[1])) 