import os
from pathlib import Path
import re

DOC_DIRS = [Path('docs')]
ROOT_DOCS = ['project_plan.md', 'comprehensive_project_guide.md']

REPLACEMENTS = {
    r"Claude 3\.7 Sonnet": "Cursor AI 및 Codex CLI",
    r"Claude 3\.7": "Cursor AI 및 Codex CLI",
    r"Claude": "Cursor AI 및 Codex CLI",
}

NOTE_LINE = (
    '> ⚠️ 본 문서에서 언급된 Claude 관련 내용은 Cursor AI 및 Codex CLI가 담당하는 것으로 변경되었습니다.'
)


def update_file(file_path: Path):
    text = file_path.read_text(encoding='utf-8')
    # Add note after top-level title if not already present
    if NOTE_LINE not in text:
        # find first heading line (starts with # )
        lines = text.split('\n')
        for idx, line in enumerate(lines):
            if line.startswith('#'):
                lines.insert(idx + 1, NOTE_LINE)
                break
        text = '\n'.join(lines)

    # Apply replacements
    for pattern, repl in REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)

    file_path.write_text(text, encoding='utf-8')
    print(f"Updated: {file_path}")


def main():
    files = []
    for doc_dir in DOC_DIRS:
        files.extend(doc_dir.rglob('*.md'))
    for root_doc in ROOT_DOCS:
        if Path(root_doc).exists():
            files.append(Path(root_doc))

    for file_path in files:
        update_file(file_path)

    print("모든 문서 업데이트 완료.")


if __name__ == "__main__":
    main() 