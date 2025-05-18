# content_generation.py
"""
Codebase content generation workflow for tutorial chapters.
"""

import os
import yaml
import re
from typing import List, Dict, Any, Tuple, Optional

from ....mcp_init import mcp
from ...atoms.codebase_knowledge import call_llm
from .code_analysis import get_content_for_indices


@mcp.workflow(
    name="write_chapters",
    description="Write tutorial chapters for a codebase using LLM"
)
async def write_chapters(
    chapter_order: List[int],
    abstractions: List[Dict[str, Any]],
    files_data: List[Tuple[str, str]],
    project_name: str,
    language: str = "english",
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Write tutorial chapters for a codebase using LLM.
    
    Args:
        chapter_order: Ordered list of abstraction indices
        abstractions: List of abstractions (from identify_abstractions)
        files_data: List of tuples containing (file_path, file_content)
        project_name: Name of the project
        language: Language for the output (default: english)
        use_cache: Whether to use cached LLM responses
        
    Returns:
        List of dictionaries containing chapter information and content
    """
    # Generate filename mapping for each chapter
    chapter_filenames = {}
    all_chapters = []
    
    for i, abstraction_index in enumerate(chapter_order):
        chapter_num = i + 1
        chapter_name = abstractions[abstraction_index]["name"]
        
        # Create safe filename from chapter name
        safe_name = re.sub(r'[^\w]', '_', chapter_name).lower()
        filename = f"{i+1:02d}_{safe_name}.md"
        
        # Add to all chapters list with link
        all_chapters.append(f"{chapter_num}. [{chapter_name}]({filename})")
        
        # Store mapping for linking between chapters
        chapter_filenames[abstraction_index] = {
            "num": chapter_num,
            "name": chapter_name,
            "filename": filename
        }
    
    # Format full chapter listing
    full_chapter_listing = "\n".join(all_chapters)
    
    # Store written chapters for context in subsequent chapters
    chapters_written_so_far = []
    chapter_results = []
    
    # Process each chapter in order
    for i, abstraction_index in enumerate(chapter_order):
        # Get abstraction details
        abstraction_details = abstractions[abstraction_index]
        chapter_num = i + 1
        
        # Get related file content
        related_file_indices = abstraction_details.get("files", [])
        related_files_content_map = get_content_for_indices(files_data, related_file_indices)
        
        # Create file context string
        file_context_str = "\n\n".join(
            f"--- File: {idx_path.split('# ')[1] if '# ' in idx_path else idx_path} ---\n{content}"
            for idx_path, content in related_files_content_map.items()
        )
        
        # Get previous and next chapter info for transitions
        prev_chapter = None
        if i > 0:
            prev_idx = chapter_order[i - 1]
            prev_chapter = chapter_filenames[prev_idx]
        
        next_chapter = None
        if i < len(chapter_order) - 1:
            next_idx = chapter_order[i + 1]
            next_chapter = chapter_filenames[next_idx]
        
        # Get summary of chapters written so far
        previous_chapters_summary = "\n---\n".join(chapters_written_so_far)
        
        # Add language instruction and context notes only if not English
        language_instruction = ""
        concept_details_note = ""
        structure_note = ""
        prev_summary_note = ""
        instruction_lang_note = ""
        mermaid_lang_note = ""
        code_comment_note = ""
        link_lang_note = ""
        tone_note = ""
        
        if language.lower() != "english":
            lang_cap = language.capitalize()
            language_instruction = f"IMPORTANT: Write this ENTIRE tutorial chapter in **{lang_cap}**. Some input context (like concept name, description, chapter list, previous summary) might already be in {lang_cap}, but you MUST translate ALL other generated content including explanations, examples, technical terms, and potentially code comments into {lang_cap}. DO NOT use English anywhere except in code syntax, required proper nouns, or when specified. The entire output MUST be in {lang_cap}.\n\n"
            concept_details_note = f" (Note: Provided in {lang_cap})"
            structure_note = f" (Note: Chapter names might be in {lang_cap})"
            prev_summary_note = f" (Note: This summary might be in {lang_cap})"
            instruction_lang_note = f" (in {lang_cap})"
            mermaid_lang_note = f" (Use {lang_cap} for labels/text if appropriate)"
            code_comment_note = f" (Translate to {lang_cap} if possible, otherwise keep minimal English for clarity)"
            link_lang_note = f" (Use the {lang_cap} chapter title from the structure above)"
            tone_note = f" (appropriate for {lang_cap} readers)"
        
        prompt = f"""
{language_instruction}Write a very beginner-friendly tutorial chapter (in Markdown format) for the project `{project_name}` about the concept: "{abstraction_details['name']}". This is Chapter {chapter_num}.

Concept Details{concept_details_note}:
- Name: {abstraction_details['name']}
- Description:
{abstraction_details['description']}

Complete Tutorial Structure{structure_note}:
{full_chapter_listing}

Context from previous chapters{prev_summary_note}:
{previous_chapters_summary if previous_chapters_summary else "This is the first chapter."}

Relevant Code Snippets (Code itself remains unchanged):
{file_context_str if file_context_str else "No specific code snippets provided for this abstraction."}

Instructions for the chapter (Generate content in {language.capitalize()} unless specified otherwise):
- Start with a clear heading (e.g., `# Chapter {chapter_num}: {abstraction_details['name']}`). Use the provided concept name.

- If this is not the first chapter, begin with a brief transition from the previous chapter{instruction_lang_note}, referencing it with a proper Markdown link using its name{link_lang_note}.

- Begin with a high-level motivation explaining what problem this abstraction solves{instruction_lang_note}. Start with a central use case as a concrete example. The whole chapter should guide the reader to understand how to solve this use case. Make it very minimal and friendly to beginners.

- If the abstraction is complex, break it down into key concepts. Explain each concept one-by-one in a very beginner-friendly way{instruction_lang_note}.

- Explain how to use this abstraction to solve the use case{instruction_lang_note}. Give example inputs and outputs for code snippets (if the output isn't values, describe at a high level what will happen{instruction_lang_note}).

- Each code block should be BELOW 10 lines! If longer code blocks are needed, break them down into smaller pieces and walk through them one-by-one. Aggresively simplify the code to make it minimal. Use comments{code_comment_note} to skip non-important implementation details. Each code block should have a beginner friendly explanation right after it{instruction_lang_note}.

- Describe the internal implementation to help understand what's under the hood{instruction_lang_note}. First provide a non-code or code-light walkthrough on what happens step-by-step when the abstraction is called{instruction_lang_note}. It's recommended to use a simple sequenceDiagram with a dummy example - keep it minimal with at most 5 participants to ensure clarity. If participant name has space, use: `participant QP as Query Processing`. {mermaid_lang_note}.

- Then dive deeper into code for the internal implementation with references to files. Provide example code blocks, but make them similarly simple and beginner-friendly. Explain{instruction_lang_note}.

- IMPORTANT: When you need to refer to other core abstractions covered in other chapters, ALWAYS use proper Markdown links like this: [Chapter Title](filename.md). Use the Complete Tutorial Structure above to find the correct filename and the chapter title{link_lang_note}. Translate the surrounding text.

- Use mermaid diagrams to illustrate complex concepts (```mermaid``` format). {mermaid_lang_note}.

- Heavily use analogies and examples throughout{instruction_lang_note} to help beginners understand.

- End the chapter with a brief conclusion that summarizes what was learned{instruction_lang_note} and provides a transition to the next chapter{instruction_lang_note}. If there is a next chapter, use a proper Markdown link: [Next Chapter Title](next_chapter_filename){link_lang_note}.

- Ensure the tone is welcoming and easy for a newcomer to understand{tone_note}.

- Output *only* the Markdown content for this chapter.

Now, directly provide a super beginner-friendly Markdown output (DON'T need ```markdown``` tags):
"""
        
        # Call the LLM
        chapter_content = await call_llm(prompt, use_cache=use_cache)
        
        # Basic validation/cleanup of the heading
        actual_heading = f"# Chapter {chapter_num}: {abstraction_details['name']}"
        if not chapter_content.strip().startswith(f"# Chapter {chapter_num}"):
            # Add or replace heading if missing or incorrect
            lines = chapter_content.strip().split("\n")
            if lines and lines[0].strip().startswith("#"):  # If there's some heading, replace it
                lines[0] = actual_heading
                chapter_content = "\n".join(lines)
            else:  # Otherwise, prepend it
                chapter_content = f"{actual_heading}\n\n{chapter_content}"
        
        # Add chapter info to results
        chapter_info = {
            "index": abstraction_index,
            "chapter_num": chapter_num,
            "title": abstraction_details["name"],
            "filename": chapter_filenames[abstraction_index]["filename"],
            "content": chapter_content,
            "prev_chapter": prev_chapter,
            "next_chapter": next_chapter
        }
        
        chapter_results.append(chapter_info)
        
        # Add to written chapters for context
        chapters_written_so_far.append(chapter_content)
    
    return chapter_results


@mcp.workflow(
    name="combine_tutorial",
    description="Combine tutorial chapters into structured files and generate index"
)
async def combine_tutorial(
    chapter_results: List[Dict[str, Any]],
    abstractions: List[Dict[str, Any]],
    relationships: Dict[str, Any],
    project_name: str,
    output_dir: str = "output",
    repo_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Combine generated chapters into structured output files.
    
    Args:
        chapter_results: List of chapter information and content (from write_chapters)
        abstractions: List of abstractions
        relationships: Dictionary with relationship details
        project_name: Name of the project
        output_dir: Output directory for the tutorial
        repo_url: Optional repository URL
        
    Returns:
        Dictionary with output information
    """
    # Prepare output directory
    project_output_dir = os.path.join(output_dir, project_name)
    os.makedirs(project_output_dir, exist_ok=True)
    
    # Generate Mermaid diagram for relationships
    mermaid_lines = ["flowchart TD"]
    
    # Add nodes for each abstraction using potentially translated names
    for i, abstr in enumerate(abstractions):
        node_id = f"A{i}"
        # Sanitize for Mermaid ID and label
        sanitized_name = abstr["name"].replace('"', "")
        mermaid_lines.append(f'    {node_id}["{sanitized_name}"]')
    
    # Add edges for relationships using potentially translated labels
    for rel in relationships["details"]:
        from_node_id = f"A{rel['from']}"
        to_node_id = f"A{rel['to']}"
        # Sanitize edge label
        edge_label = rel["label"].replace('"', "").replace("\n", " ")
        max_label_len = 30
        if len(edge_label) > max_label_len:
            edge_label = edge_label[: max_label_len - 3] + "..."
        mermaid_lines.append(f'    {from_node_id} -- "{edge_label}" --> {to_node_id}')
    
    # Create full mermaid diagram
    mermaid_diagram = "\n".join(mermaid_lines)
    
    # Generate index.md content
    index_content = f"# Tutorial: {project_name}\n\n"
    index_content += f"{relationships['summary']}\n\n"
    
    if repo_url:
        index_content += f"**Source Repository:** [{repo_url}]({repo_url})\n\n"
    
    # Add Mermaid diagram for relationships
    index_content += "```mermaid\n"
    index_content += mermaid_diagram + "\n"
    index_content += "```\n\n"
    
    index_content += f"## Chapters\n\n"
    
    # Add links to chapters
    for chapter in sorted(chapter_results, key=lambda x: x["chapter_num"]):
        index_content += f"{chapter['chapter_num']}. [{chapter['title']}]({chapter['filename']})\n"
    
    # Add attribution
    index_content += f"\n\n---\n\nGenerated by Palantir Codebase Knowledge Builder"
    
    # Write index.md
    index_path = os.path.join(project_output_dir, "index.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)
    
    # Write chapter files
    for chapter in chapter_results:
        # Add attribution to chapter content
        chapter_content = chapter["content"]
        if not chapter_content.endswith("\n\n"):
            chapter_content += "\n\n"
        chapter_content += f"---\n\nGenerated by Palantir Codebase Knowledge Builder"
        
        # Write the chapter file
        chapter_path = os.path.join(project_output_dir, chapter["filename"])
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(chapter_content)
    
    return {
        "output_dir": project_output_dir,
        "index_path": index_path,
        "chapter_files": [os.path.join(project_output_dir, chapter["filename"]) for chapter in chapter_results]
    }
