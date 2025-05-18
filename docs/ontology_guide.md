# Ontology Guide — Education Domain (v1.0)

[Ontology Quick Start](quick_start_ontology.md)

## 1. Core Object Types (`:MetaObject`)
| Label | Description | Key Properties |
|-------|-------------|----------------|
| `Student` | 학습자 | `student_id (PK)`, `name`, `grade_level`, `cohort` |
| `Course`  | 과목/강좌 | `course_code (PK)`, `title`, `credits`, `domain` |
| `Topic`   | 학습 개념 | `topic_id (PK)`, `name_kr`, `name_en`, `difficulty` |
| `Instructor` | 강사 | `instructor_id (PK)`, `name`, `department` |
| `Assessment` | 시험·퀴즈 | `assessment_id (PK)`, `type`, `max_score` |

## 2. Relationship Types (`:MetaRel`)
| Type | Domain ➜ Range | Semantics |
|------|----------------|-----------|
| `ENROLLED_IN` | `Student ➜ Course` | 수강 등록 |
| `TEACHES` | `Instructor ➜ Course` | 강의 담당 |
| `COVERS` | `Course ➜ Topic` | 과목-토픽 매핑 |
| `PREREQ_OF` | `Topic ➜ Topic` | 선수 개념 |
| `ASSESSED_BY` | `Student ➜ Assessment` | 시험 응시 |
| `MEASURES` | `Assessment ➜ Topic` | 시험-토픽 연계 |

## 3. Naming & Versioning
```cypher
MERGE (m:MetaObject {name:'Course'})
ON CREATE SET m.version = '1.0.0'
ON MATCH  SET m.version = '1.0.1'  // bumped by pipeline
```

## 4. Cypher Snippets — Cursor AI·Codex Ready

```cypher
// Load students & link to courses
LOAD CSV WITH HEADERS FROM 'file:///students.csv' AS row
MERGE (s:Student {student_id: row.id})
SET   s += row
WITH s, row
MATCH (c:Course {course_code: row.course})
MERGE (s)-[:ENROLLED_IN]->(c);
```

## 5. Graph Algorithms (Neo4j GDS)

```python
# analysis/notebooks/prereq_pagerank.py
"""
Compute PageRank on Topic prerequisite graph.
Outputs: topic_pagerank.csv
"""
import pandas as pd
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","pass"))
with driver.session() as s:
    s.run("CALL gds.graph.project('topicGraph','Topic',{PREREQ_OF:{orientation:'NATURAL'}})")
    r = s.run("""
        CALL gds.pageRank.stream('topicGraph')
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).name_kr AS topic, score
        ORDER BY score DESC
    """)
    pd.DataFrame(r.data()).to_csv("output/topic_pagerank.csv", index=False)
``` 