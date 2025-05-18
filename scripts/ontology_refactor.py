"""ontology_refactor.py
Codex CLI를 호출해 CSV 스키마 diff로부터 Cypher 패치를 생성·실행한다.

Usage:
    poetry run python scripts/ontology_refactor.py diff.csv
전제:
1) `ai_resources/prompts/ontology_refactor_prompt.md` 가 프롬프트 템플릿으로 존재
2) codex-cli 가 설치돼 있고 환경변수 CODEX_API_KEY 설정되어 있음
"""
import subprocess
import sys
from pathlib import Path

PROMPT_FILE = Path("ai_resources/prompts/ontology_refactor_prompt.md")

def run_codex(diff_path: Path):
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(PROMPT_FILE)
    command = [
        "codex", "run", "--prompt", str(PROMPT_FILE), "--var", f"csv_diff_path={diff_path}", "--output", "ontology_patch.cypher"
    ]
    print("🛠️  Running Codex CLI to generate Cypher patch ...")
    subprocess.run(command, check=True)


def apply_cypher_patch():
    patch_file = Path("ontology_patch.cypher")
    if not patch_file.exists():
        raise FileNotFoundError(patch_file)
    print("🔗 Applying Cypher patch to Neo4j ...")
    subprocess.run(["codex", "run", "cypher", str(patch_file)], check=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/ontology_refactor.py <csv_diff_path>")
        sys.exit(1)
    diff_path = Path(sys.argv[1])
    if not diff_path.exists():
        print("Diff file not found")
        sys.exit(1)
    run_codex(diff_path)
    apply_cypher_patch()
    print("✅ Ontology refactor complete")


if __name__ == "__main__":
    main() 