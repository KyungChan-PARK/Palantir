# tutorial_generator.py
"""
Codebase Knowledge Tutorial Generator System.
This system combines multiple workflows to generate comprehensive tutorials for codebases.
"""

import os
from typing import Dict, Any, List, Union, Set, Optional, Tuple

from ....mcp_init import mcp
from ...atoms.codebase_knowledge import crawl_github_files, crawl_local_files
from ...molecules.codebase_knowledge import (
    identify_abstractions,
    analyze_relationships,
    order_chapters,
    write_chapters,
    combine_tutorial
)

# Default include/exclude patterns for file filtering
DEFAULT_INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java", "*.pyi", "*.pyx",
    "*.c", "*.cc", "*.cpp", "*.h", "*.md", "*.rst", "Dockerfile",
    "Makefile", "*.yaml", "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "assets/*", "data/*", "examples/*", "images/*", "public/*", "static/*", "temp/*",
    "docs/*", 
    "venv/*", ".venv/*", "*test*", "tests/*", "docs/*", "examples/*",
    "dist/*", "build/*", "experimental/*", "deprecated/*", "misc/*", 
    "legacy/*", ".git/*", ".github/*", ".next/*", ".vscode/*", "obj/*", "bin/*", "node_modules/*", "*.log"
}


@mcp.system(
    name="generate_codebase_tutorial",
    description="Generate a comprehensive tutorial for a codebase from GitHub or local directory"
)
async def generate_codebase_tutorial(
    # Source parameters (exactly one must be provided)
    repo_url: Optional[str] = None,
    local_dir: Optional[str] = None,
    
    # Project information
    project_name: Optional[str] = None,
    github_token: Optional[str] = None,
    
    # File filtering parameters
    include_patterns: Union[str, Set[str], List[str]] = None,
    exclude_patterns: Union[str, Set[str], List[str]] = None,
    max_file_size: int = 100000,  # 100KB default
    
    # Output parameters
    output_dir: str = "output",
    
    # Language parameters
    language: str = "english",
    
    # Analysis parameters
    use_cache: bool = True,
    max_abstraction_num: int = 10
) -> Dict[str, Any]:
    """
    Generate a comprehensive tutorial for a codebase from GitHub or local directory.
    
    Args:
        repo_url: URL of the GitHub repository (alternative to local_dir)
        local_dir: Path to local directory (alternative to repo_url)
        project_name: Name of the project (derived from repo/dir if not provided)
        github_token: GitHub token for private repositories
        include_patterns: File patterns to include in analysis
        exclude_patterns: File patterns to exclude from analysis
        max_file_size: Maximum file size to include in analysis (bytes)
        output_dir: Directory to save the generated tutorial
        language: Language for the tutorial content
        use_cache: Whether to use cached LLM responses
        max_abstraction_num: Maximum number of abstractions to identify
        
    Returns:
        Dictionary with information about the generated tutorial
    """
    # Validate source parameters
    if not (repo_url or local_dir) or (repo_url and local_dir):
        raise ValueError("Either repo_url or local_dir must be provided, but not both")
    
    # Process include/exclude patterns
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS.copy()
    elif isinstance(include_patterns, (str, list)):
        include_patterns = set(include_patterns if isinstance(include_patterns, list) else [include_patterns])
    
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS.copy()
    elif isinstance(exclude_patterns, (str, list)):
        exclude_patterns = set(exclude_patterns if isinstance(exclude_patterns, list) else [exclude_patterns])
    
    # Determine project name if not provided
    if not project_name:
        if repo_url:
            # Extract from URL
            project_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        elif local_dir:
            # Use directory name
            project_name = os.path.basename(os.path.abspath(local_dir))
    
    print(f"Generating tutorial for project: {project_name}")
    print(f"Language: {language}")
    print(f"LLM caching: {'Enabled' if use_cache else 'Disabled'}")
    
    # 1. Crawl files from source
    if repo_url:
        print(f"Crawling GitHub repository: {repo_url}")
        result = await crawl_github_files(
            repo_url=repo_url,
            token=github_token,
            max_file_size=max_file_size,
            use_relative_paths=True,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns
        )
    else:
        print(f"Crawling local directory: {local_dir}")
        result = await crawl_local_files(
            directory=local_dir,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_file_size=max_file_size,
            use_relative_paths=True
        )
    
    # Convert dictionary to list of tuples
    files_dict = result.get("files", {})
    files_data = [(path, content) for path, content in files_dict.items()]
    
    if not files_data:
        raise ValueError("No files were found or all files were filtered out")
    
    print(f"Found {len(files_data)} files for analysis")
    
    # 2. Identify abstractions in the codebase
    print("Identifying abstractions...")
    abstractions = await identify_abstractions(
        files_data=files_data,
        project_name=project_name,
        language=language,
        use_cache=use_cache,
        max_abstraction_num=max_abstraction_num
    )
    
    print(f"Identified {len(abstractions)} abstractions")
    
    # 3. Analyze relationships between abstractions
    print("Analyzing relationships between abstractions...")
    relationships = await analyze_relationships(
        abstractions=abstractions,
        files_data=files_data,
        project_name=project_name,
        language=language,
        use_cache=use_cache
    )
    
    print(f"Analyzed relationships and generated project summary")
    
    # 4. Determine optimal chapter order
    print("Determining optimal chapter order...")
    chapter_order = await order_chapters(
        abstractions=abstractions,
        relationships=relationships,
        project_name=project_name,
        language=language,
        use_cache=use_cache
    )
    
    print(f"Determined chapter order")
    
    # 5. Write the tutorial chapters
    print("Writing tutorial chapters...")
    chapter_results = await write_chapters(
        chapter_order=chapter_order,
        abstractions=abstractions,
        files_data=files_data,
        project_name=project_name,
        language=language,
        use_cache=use_cache
    )
    
    print(f"Generated {len(chapter_results)} tutorial chapters")
    
    # 6. Combine chapters into final tutorial
    print("Combining tutorial chapters...")
    output_info = await combine_tutorial(
        chapter_results=chapter_results,
        abstractions=abstractions,
        relationships=relationships,
        project_name=project_name,
        output_dir=output_dir,
        repo_url=repo_url
    )
    
    print(f"Tutorial generation complete!")
    print(f"Output directory: {output_info['output_dir']}")
    print(f"Index file: {output_info['index_path']}")
    print(f"Generated {len(output_info['chapter_files'])} chapter files")
    
    # Return comprehensive result
    return {
        "project_name": project_name,
        "abstractions_count": len(abstractions),
        "chapters_count": len(chapter_results),
        "output_dir": output_info["output_dir"],
        "index_path": output_info["index_path"],
        "language": language,
        "sources": {
            "repo_url": repo_url,
            "local_dir": local_dir,
            "files_count": len(files_data)
        }
    }
