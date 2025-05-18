"""
테스트 문서 생성 도구

팔란티어 파운드리 시스템의 성능 테스트를 위한 테스트 문서를 생성합니다.
"""

import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from analysis.mcp_init import mcp

# 로깅 설정
logger = logging.getLogger("generate_test_documents")

# 문서 템플릿
DOCUMENT_TEMPLATES = {
    "report": {
        "title_prefix": "보고서: ",
        "sections": [
            "요약",
            "배경",
            "분석",
            "결과",
            "결론",
            "참고 문헌"
        ],
        "paragraphs_per_section": (2, 5),
        "sentences_per_paragraph": (3, 8)
    },
    "analysis": {
        "title_prefix": "분석: ",
        "sections": [
            "목적",
            "데이터",
            "방법론",
            "탐색적 분석",
            "결과 해석",
            "제한 사항",
            "향후 방향"
        ],
        "paragraphs_per_section": (2, 4),
        "sentences_per_paragraph": (4, 10)
    },
    "memo": {
        "title_prefix": "메모: ",
        "sections": [
            "개요",
            "내용",
            "조치 사항"
        ],
        "paragraphs_per_section": (1, 3),
        "sentences_per_paragraph": (2, 5)
    }
}

# 문장 생성용 샘플 텍스트
SAMPLE_SENTENCES = [
    "이 문서는 테스트 목적으로 생성되었습니다.",
    "팔란티어 파운드리는 데이터 통합 및 분석 플랫폼입니다.",
    "온톨로지를 통해 데이터 간의 관계를 정의할 수 있습니다.",
    "데이터 파이프라인은 데이터 처리 워크플로우를 자동화합니다.",
    "대시보드는 데이터 시각화 및 상호작용을 위한 인터페이스를 제공합니다.",
    "Neo4j는 그래프 데이터베이스로 복잡한 관계 모델링에 적합합니다.",
    "Apache Airflow는 워크플로우 관리 도구로 데이터 파이프라인 구축에 사용됩니다.",
    "Great Expectations는 데이터 검증 및 품질 모니터링을 지원합니다.",
    "LLM(Large Language Model)은 자연어 처리 및 코드 생성에 활용됩니다.",
    "RAG(Retrieval Augmented Generation)는 외부 지식을 활용한 텍스트 생성 방식입니다.",
    "문서 관리 시스템은 문서의 생성, 저장, 검색, 공유 기능을 제공합니다.",
    "메타데이터는 데이터에 대한 데이터로, 데이터의 속성을 설명합니다.",
    "API는 시스템 간의 인터페이스를 정의하고 통신을 가능하게 합니다.",
    "데이터 품질은 데이터의 정확성, 완전성, 일관성, 유효성을 보장합니다.",
    "MCP(Model Context Protocol)는 모델과 컨텍스트 간의 상호작용을 정의합니다.",
    "시스템 성능은 처리 속도, 응답 시간, 처리량 등으로 측정됩니다.",
    "최적화는 시스템의 효율성과 성능을 향상시키는 과정입니다.",
    "컨텍스트는 상황이나 환경에 대한 정보를 제공하는 데이터입니다.",
    "설계 패턴은 소프트웨어 설계에서 반복적으로 발생하는 문제에 대한 해결책입니다.",
    "모듈식 설계는 시스템을 독립적인 모듈로 분리하여 유지보수성을 높입니다."
]

@mcp.tool(
    name="create_test_document_set",
    description="테스트 문서 세트 생성 도구",
    tags=["test", "document", "generation"]
)
async def create_test_document_set(
    output_dir: str,
    count: int = 10,
    distribution: Optional[Dict[str, float]] = None,
    metadata_file: Optional[str] = None
) -> Dict[str, Union[int, str]]:
    """테스트 문서 세트 생성
    
    Args:
        output_dir: 문서를 저장할 디렉토리 경로
        count: 생성할 문서 수
        distribution: 문서 유형별 분포 (예: {"report": 0.5, "analysis": 0.3, "memo": 0.2})
        metadata_file: 메타데이터를 저장할 파일 경로
        
    Returns:
        생성 결과 정보
    """
    try:
        # 기본 분포 설정
        if distribution is None:
            distribution = {"report": 0.4, "analysis": 0.4, "memo": 0.2}
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 메타데이터 저장용 리스트
        metadata_list = []
        
        logger.info(f"테스트 문서 {count}개 생성 중...")
        
        # 문서 생성
        for i in range(count):
            # 문서 유형 선택
            doc_type = random.choices(
                list(distribution.keys()),
                weights=list(distribution.values()),
                k=1
            )[0]
            
            # 문서 생성
            document, metadata = await generate_document(doc_type, i)
            
            # 문서 저장
            filename = f"{doc_type}_{i:04d}.txt"
            file_path = os.path.join(output_dir, filename)
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(document)
            
            # 메타데이터 추가
            metadata["file_path"] = file_path
            metadata["filename"] = filename
            metadata_list.append(metadata)
            
            logger.debug(f"문서 생성 완료: {filename}")
        
        # 메타데이터 저장
        if metadata_file:
            with open(metadata_file, "w", encoding="utf-8") as file:
                json.dump(metadata_list, file, indent=2, ensure_ascii=False)
        
        logger.info(f"테스트 문서 {count}개 생성 완료: {output_dir}")
        
        return {
            "count": count,
            "output_dir": output_dir,
            "metadata_file": metadata_file,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"테스트 문서 생성 오류: {e}")
        raise

async def generate_document(doc_type: str, index: int) -> tuple:
    """문서 및 메타데이터 생성
    
    Args:
        doc_type: 문서 유형 ("report", "analysis", "memo" 중 하나)
        index: 문서 인덱스
        
    Returns:
        (생성된 문서 텍스트, 메타데이터 딕셔너리) 튜플
    """
    template = DOCUMENT_TEMPLATES[doc_type]
    
    # 제목 생성
    topics = ["데이터 분석", "시스템 성능", "온톨로지 설계", "파이프라인 최적화", 
             "대시보드 개발", "API 통합", "데이터 품질", "LLM 활용", "문서 관리"]
    title = f"{template['title_prefix']}{random.choice(topics)} #{index}"
    
    # 문서 내용 생성
    document_parts = [f"# {title}\n\n"]
    
    # 메타데이터 준비
    created_at = datetime.now() - timedelta(days=random.randint(0, 30))
    doc_id = str(uuid.uuid4())
    
    # 섹션 생성
    for section in template["sections"]:
        document_parts.append(f"## {section}\n\n")
        
        # 섹션 내 문단 생성
        num_paragraphs = random.randint(
            template["paragraphs_per_section"][0],
            template["paragraphs_per_section"][1]
        )
        
        for _ in range(num_paragraphs):
            # 문단 내 문장 생성
            num_sentences = random.randint(
                template["sentences_per_paragraph"][0],
                template["sentences_per_paragraph"][1]
            )
            
            sentences = random.choices(SAMPLE_SENTENCES, k=num_sentences)
            paragraph = " ".join(sentences)
            document_parts.append(f"{paragraph}\n\n")
    
    # 메타데이터 생성
    metadata = {
        "doc_id": doc_id,
        "title": title,
        "doc_type": doc_type,
        "created_at": created_at.isoformat(),
        "sections": len(template["sections"]),
        "status": random.choice(["draft", "review", "approved", "published", "archived"])
    }
    
    return "".join(document_parts), metadata

# 직접 실행 시
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    import asyncio
    
    async def main():
        result = await create_test_document_set(
            output_dir="C:\\Users\\packr\\OneDrive\\palantir\\temp\\test_documents\\docs_sample",
            count=5,
            metadata_file="C:\\Users\\packr\\OneDrive\\palantir\\temp\\test_documents\\metadata_sample.json"
        )
        print(f"결과: {result}")
    
    asyncio.run(main())
