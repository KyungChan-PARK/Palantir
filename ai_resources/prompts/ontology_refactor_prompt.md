# Prompt — Ontology Refactor (Codex CLI)

**Goal** – Given a CSV schema diff, output Cypher patch to  
1) Create/alter MetaObjects & properties  
2) Update `PREREQ_OF` links  
3) Bump `:.version`

```yaml
input:
  csv_diff: |
    - old_schema.csv
    + new_schema.csv
tasks:
  - parse_diff
  - detect_object_changes
  - write_cypher_patch
validate:
  - property_types_match
  - metaobject_versions_incremented
``` 