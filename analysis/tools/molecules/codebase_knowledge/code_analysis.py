# code_analysis.py
"""
Codebase analysis workflows for:
1. Identifying key abstractions in a codebase
2. Analyzing relationships between abstractions
3. Determining optimal chapter order for a tutorial
"""

import os
import yaml
from typing import List, Dict, Any, Union, Set, Tuple

from ....mcp_init import mcp
from ...atoms.codebase_knowledge import call_llm

# Helper functions
def get_content_for_indices(files_data, indices):
    """Get file content for specific indices from files_data."""
    content_map = {}
    for i in indices:
        if isinstance(files_data, list) and 0 <= i < len(files_data):
            path, content = files_data[i]
            content_map[f"{i} # {path}"] = content
        elif isinstance(files_data, dict) and str(i) in files_data:
            # Handle dictionary format
            content_map[f"{i} # {str(i)}"] = files_data[str(i)]
    return content_map


@mcp.workflow(
    name="identify_abstractions",
    description="Identify core abstractions in a codebase using LLM"
)
async def identify_abstractions(
    files_data: List[Tuple[str, str]], 
    project_name: str,
    language: str = "english",
    use_cache: bool = True,
    max_abstraction_num: int = 10
) -> List[Dict[str, Any]]:
    """
    Identify core abstractions in a codebase using LLM.
    
    Args:
        files_data: List of tuples containing (file_path, file_content)
        project_name: Name of the project
        language: Language for the output (default: english)
        use_cache: Whether to use cached LLM responses
        max_abstraction_num: Maximum number of abstractions to identify
        
    Returns:
        List of abstraction details containing name, description, and relevant file indices
    """
    # Create context from files
    context = ""
    file_info = []  # Store tuples of (index, path)
    
    for i, (path, content) in enumerate(files_data):
        entry = f"--- File Index {i}: {path} ---\n{content}\n\n"
        context += entry
        file_info.append((i, path))
    
    # Format file info for the prompt
    file_listing_for_prompt = "\n".join([f"- {idx} # {path}" for idx, path in file_info])
    
    # Add language instruction and hints only if not English
    language_instruction = ""
    name_lang_hint = ""
    desc_lang_hint = ""
    if language.lower() != "english":
        language_instruction = f"IMPORTANT: Generate the `name` and `description` for each abstraction in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
        # Keep specific hints here as name/description are primary targets
        name_lang_hint = f" (value in {language.capitalize()})"
        desc_lang_hint = f" (value in {language.capitalize()})"
    
    prompt = f"""
For the project `{project_name}`:

Codebase Context:
{context}

{language_instruction}Analyze the codebase context.
Identify the top 5-{max_abstraction_num} core most important abstractions to help those new to the codebase.

For each abstraction, provide:
1. A concise `name`{name_lang_hint}.
2. A beginner-friendly `description` explaining what it is with a simple analogy, in around 100 words{desc_lang_hint}.
3. A list of relevant `file_indices` (integers) using the format `idx # path/comment`.

List of file indices and paths present in the context:
{file_listing_for_prompt}

Format the output as a YAML list of dictionaries:

```yaml
- name: |
    Query Processing{name_lang_hint}
  description: |
    Explains what the abstraction does.
    It's like a central dispatcher routing requests.{desc_lang_hint}
  file_indices:
    - 0 # path/to/file1.py
    - 3 # path/to/related.py
- name: |
    Query Optimization{name_lang_hint}
  description: |
    Another core concept, similar to a blueprint for objects.{desc_lang_hint}
  file_indices:
    - 5 # path/to/another.js
# ... up to {max_abstraction_num} abstractions
```"""

    # Call the LLM
    response = await call_llm(prompt, use_cache=use_cache)
    
    # Extract YAML content from response
    yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
    abstractions = yaml.safe_load(yaml_str)
    
    # Validate the output
    if not isinstance(abstractions, list):
        raise ValueError("LLM Output is not a list")
    
    # Process and validate each abstraction
    validated_abstractions = []
    for item in abstractions:
        if not isinstance(item, dict) or not all(
            k in item for k in ["name", "description", "file_indices"]
        ):
            raise ValueError(f"Missing keys in abstraction item: {item}")
        
        # Validate indices
        validated_indices = []
        for idx_entry in item["file_indices"]:
            try:
                if isinstance(idx_entry, int):
                    idx = idx_entry
                elif isinstance(idx_entry, str) and "#" in idx_entry:
                    idx = int(idx_entry.split("#")[0].strip())
                else:
                    idx = int(str(idx_entry).strip())

                if not (0 <= idx < len(files_data)):
                    raise ValueError(
                        f"Invalid file index {idx} found in item {item['name']}. Max index is {len(files_data) - 1}."
                    )
                validated_indices.append(idx)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Could not parse index from entry: {idx_entry} in item {item['name']}"
                )
        
        # Store validated abstraction with normalized structure
        validated_abstractions.append({
            "name": item["name"],  # Potentially translated name
            "description": item["description"],  # Potentially translated description
            "files": sorted(list(set(validated_indices)))  # Unique, sorted file indices
        })
    
    return validated_abstractions


@mcp.workflow(
    name="analyze_relationships",
    description="Analyze relationships between codebase abstractions using LLM"
)
async def analyze_relationships(
    abstractions: List[Dict[str, Any]], 
    files_data: List[Tuple[str, str]],
    project_name: str,
    language: str = "english",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Analyze relationships between codebase abstractions using LLM.
    
    Args:
        abstractions: List of abstractions (from identify_abstractions)
        files_data: List of tuples containing (file_path, file_content)
        project_name: Name of the project
        language: Language for the output (default: english)
        use_cache: Whether to use cached LLM responses
        
    Returns:
        Dictionary with project summary and relationship details
    """
    # Create context with abstraction information
    context = "Identified Abstractions:\n"
    all_relevant_indices = set()
    abstraction_info_for_prompt = []
    
    for i, abstr in enumerate(abstractions):
        file_indices_str = ", ".join(map(str, abstr["files"]))
        info_line = (
            f"- Index {i}: {abstr['name']} (Relevant file indices: [{file_indices_str}])\n"
            f"  Description: {abstr['description']}"
        )
        context += info_line + "\n"
        abstraction_info_for_prompt.append(f"{i} # {abstr['name']}")  # Use potentially translated name
        all_relevant_indices.update(abstr["files"])

    abstraction_listing = "\n".join(abstraction_info_for_prompt)
    
    # Add file content information to context
    context += "\nRelevant File Snippets (Referenced by Index and Path):\n"
    relevant_files_content_map = get_content_for_indices(files_data, sorted(list(all_relevant_indices)))
    
    file_context_str = "\n\n".join(
        f"--- File: {idx_path} ---\n{content}"
        for idx_path, content in relevant_files_content_map.items()
    )
    context += file_context_str
    
    # Add language instruction and hints only if not English
    language_instruction = ""
    lang_hint = ""
    list_lang_note = ""
    if language.lower() != "english":
        language_instruction = f"IMPORTANT: Generate the `summary` and relationship `label` fields in **{language.capitalize()}** language. Do NOT use English for these fields.\n\n"
        lang_hint = f" (in {language.capitalize()})"
        list_lang_note = f" (Names might be in {language.capitalize()})"  # Note for the input list
    
    prompt = f"""
Based on the following abstractions and relevant code snippets from the project `{project_name}`:

List of Abstraction Indices and Names{list_lang_note}:
{abstraction_listing}

Context (Abstractions, Descriptions, Code):
{context}

{language_instruction}Please provide:
1. A high-level `summary` of the project's main purpose and functionality in a few beginner-friendly sentences{lang_hint}. Use markdown formatting with **bold** and *italic* text to highlight important concepts.
2. A list (`relationships`) describing the key interactions between these abstractions. For each relationship, specify:
    - `from_abstraction`: Index of the source abstraction (e.g., `0 # AbstractionName1`)
    - `to_abstraction`: Index of the target abstraction (e.g., `1 # AbstractionName2`)
    - `label`: A brief label for the interaction **in just a few words**{lang_hint} (e.g., "Manages", "Inherits", "Uses").
    Ideally the relationship should be backed by one abstraction calling or passing parameters to another.
    Simplify the relationship and exclude those non-important ones.

IMPORTANT: Make sure EVERY abstraction is involved in at least ONE relationship (either as source or target). Each abstraction index must appear at least once across all relationships.

Format the output as YAML:

```yaml
summary: |
  A brief, simple explanation of the project{lang_hint}.
  Can span multiple lines with **bold** and *italic* for emphasis.
relationships:
  - from_abstraction: 0 # AbstractionName1
    to_abstraction: 1 # AbstractionName2
    label: "Manages"{lang_hint}
  - from_abstraction: 2 # AbstractionName3
    to_abstraction: 0 # AbstractionName1
    label: "Provides config"{lang_hint}
  # ... other relationships
```

Now, provide the YAML output:
"""
    
    # Call the LLM
    response = await call_llm(prompt, use_cache=use_cache)
    
    # Extract YAML content from response
    yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
    relationships_data = yaml.safe_load(yaml_str)
    
    # Validate output structure
    if not isinstance(relationships_data, dict) or not all(
        k in relationships_data for k in ["summary", "relationships"]
    ):
        raise ValueError("LLM output is not a dict or missing keys ('summary', 'relationships')")
    
    # Validate summary and relationships
    if not isinstance(relationships_data["summary"], str):
        raise ValueError("summary is not a string")
    if not isinstance(relationships_data["relationships"], list):
        raise ValueError("relationships is not a list")
    
    # Process and validate each relationship
    validated_relationships = []
    num_abstractions = len(abstractions)
    
    for rel in relationships_data["relationships"]:
        # Check structure
        if not isinstance(rel, dict) or not all(
            k in rel for k in ["from_abstraction", "to_abstraction", "label"]
        ):
            raise ValueError(
                f"Missing keys in relationship item: {rel}"
            )
        
        # Validate label is a string
        if not isinstance(rel["label"], str):
            raise ValueError(f"Relationship label is not a string: {rel}")
        
        # Extract indices
        try:
            from_idx = int(str(rel["from_abstraction"]).split("#")[0].strip())
            to_idx = int(str(rel["to_abstraction"]).split("#")[0].strip())
            
            # Validate indices are in range
            if not (0 <= from_idx < num_abstractions and 0 <= to_idx < num_abstractions):
                raise ValueError(
                    f"Invalid index in relationship: from={from_idx}, to={to_idx}. Max index is {num_abstractions-1}."
                )
            
            # Add validated relationship
            validated_relationships.append({
                "from": from_idx,
                "to": to_idx,
                "label": rel["label"],  # Potentially translated label
            })
        except (ValueError, TypeError):
            raise ValueError(f"Could not parse indices from relationship: {rel}")
    
    # Return structured relationship data
    return {
        "summary": relationships_data["summary"],  # Potentially translated summary
        "details": validated_relationships,  # Validated relationships
    }


@mcp.workflow(
    name="order_chapters",
    description="Determine optimal chapter order for a codebase tutorial"
)
async def order_chapters(
    abstractions: List[Dict[str, Any]],
    relationships: Dict[str, Any],
    project_name: str,
    language: str = "english",
    use_cache: bool = True
) -> List[int]:
    """
    Determine the optimal chapter order for a codebase tutorial.
    
    Args:
        abstractions: List of abstractions (from identify_abstractions)
        relationships: Dictionary with relationship details (from analyze_relationships)
        project_name: Name of the project
        language: Language for the output (default: english)
        use_cache: Whether to use cached LLM responses
        
    Returns:
        Ordered list of abstraction indices for chapters
    """
    # Prepare abstraction info for the prompt
    abstraction_info_for_prompt = []
    for i, a in enumerate(abstractions):
        abstraction_info_for_prompt.append(f"- {i} # {a['name']}")  # Use potentially translated name
    abstraction_listing = "\n".join(abstraction_info_for_prompt)
    
    # Note about language if non-English
    summary_note = ""
    if language.lower() != "english":
        summary_note = f" (Note: Project Summary might be in {language.capitalize()})"
    
    # Prepare context with relationships information
    context = f"Project Summary{summary_note}:\n{relationships['summary']}\n\n"
    context += "Relationships (Indices refer to abstractions above):\n"
    
    for rel in relationships["details"]:
        from_name = abstractions[rel["from"]]["name"]
        to_name = abstractions[rel["to"]]["name"]
        context += f"- From {rel['from']} ({from_name}) to {rel['to']} ({to_name}): {rel['label']}\n"
    
    # Add language note if non-English
    list_lang_note = ""
    if language.lower() != "english":
        list_lang_note = f" (Names might be in {language.capitalize()})"
    
    prompt = f"""
Given the following project abstractions and their relationships for the project '{project_name}':

Abstractions (Index # Name){list_lang_note}:
{abstraction_listing}

Context about relationships and project summary:
{context}

If you are going to make a tutorial for '{project_name}', what is the best order to explain these abstractions, from first to last?
Ideally, first explain those that are the most important or foundational, perhaps user-facing concepts or entry points. Then move to more detailed, lower-level implementation details or supporting concepts.

Output the ordered list of abstraction indices, including the name in a comment for clarity. Use the format `idx # AbstractionName`.

```yaml
- 2 # FoundationalConcept
- 0 # CoreClassA
- 1 # CoreClassB (uses CoreClassA)
- ...
```

Now, provide the YAML output:
"""
    
    # Call the LLM
    response = await call_llm(prompt, use_cache=use_cache)
    
    # Extract YAML content from response
    yaml_str = response.strip().split("```yaml")[1].split("```")[0].strip()
    ordered_indices_raw = yaml.safe_load(yaml_str)
    
    # Validate output structure
    if not isinstance(ordered_indices_raw, list):
        raise ValueError("LLM output is not a list")
    
    # Process and validate each index in the order
    ordered_indices = []
    seen_indices = set()
    num_abstractions = len(abstractions)
    
    for entry in ordered_indices_raw:
        try:
            # Extract index from entry
            if isinstance(entry, int):
                idx = entry
            elif isinstance(entry, str) and "#" in entry:
                idx = int(entry.split("#")[0].strip())
            else:
                idx = int(str(entry).strip())

            # Validate index is in range
            if not (0 <= idx < num_abstractions):
                raise ValueError(
                    f"Invalid index {idx} in ordered list. Max index is {num_abstractions-1}."
                )
            
            # Check for duplicates
            if idx in seen_indices:
                raise ValueError(f"Duplicate index {idx} found in ordered list.")
            
            # Add to ordered list and mark as seen
            ordered_indices.append(idx)
            seen_indices.add(idx)
        except (ValueError, TypeError):
            raise ValueError(f"Could not parse index from ordered list entry: {entry}")
    
    # Ensure all abstractions are included
    if len(ordered_indices) != num_abstractions:
        raise ValueError(
            f"Ordered list length ({len(ordered_indices)}) does not match number of abstractions ({num_abstractions}). Missing indices: {set(range(num_abstractions)) - seen_indices}"
        )
    
    return ordered_indices
