# run_codebase_tutorial.py
"""
Python script to run the codebase knowledge tutorial generator.
This is called by the run_codebase_tutorial.bat file.
"""

import os
import sys
import asyncio
import argparse
from typing import Optional, Set

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the generator
from analysis.tools.organisms.codebase_knowledge import generate_codebase_tutorial


async def run_generator(
    repo_url: Optional[str] = None,
    local_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    github_token: Optional[str] = None,
    language: str = "english",
    output_dir: str = "output",
    max_abstractions: int = 10,
    use_cache: bool = True
):
    """Run the codebase tutorial generator with the given parameters."""
    try:
        result = await generate_codebase_tutorial(
            repo_url=repo_url,
            local_dir=local_dir,
            project_name=project_name,
            github_token=github_token,
            language=language,
            output_dir=output_dir,
            max_abstraction_num=max_abstractions,
            use_cache=use_cache
        )
        
        # Print success message
        print("\nTutorial generation successful!")
        print(f"Project: {result['project_name']}")
        print(f"Abstractions: {result['abstractions_count']}")
        print(f"Chapters: {result['chapters_count']}")
        print(f"Language: {result['language']}")
        print(f"Output directory: {result['output_dir']}")
        print(f"Index file: {result['index_path']}")
        
        return 0
    except Exception as e:
        print(f"\nError generating tutorial: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main(*args):
    """Main function to parse command line arguments and run the generator."""
    parser = argparse.ArgumentParser(description="Generate a codebase knowledge tutorial")
    
    # Source arguments - mutually exclusive
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo_url", help="URL of the GitHub repository")
    source_group.add_argument("--local_dir", help="Path to local directory")
    
    # Optional arguments
    parser.add_argument("--project_name", help="Name of the project (derived from source if not provided)")
    parser.add_argument("--github_token", help="GitHub token for private repositories")
    parser.add_argument("--language", default="english", help="Language for the tutorial (default: english)")
    parser.add_argument("--output_dir", default="output", help="Output directory for the tutorial")
    parser.add_argument("--max_abstractions", type=int, default=10, help="Maximum number of abstractions to identify")
    
    # Cache control - mutually exclusive
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--use_cache", action="store_true", help="Use cached LLM responses (default)")
    cache_group.add_argument("--no_cache", action="store_true", help="Do not use cached LLM responses")
    
    # Parse arguments from command line or passed directly
    if args and len(args) > 0:
        # Convert to list if received as a string
        if len(args) == 1 and isinstance(args[0], str):
            import shlex
            args = shlex.split(args[0])
        parsed_args = parser.parse_args(args)
    else:
        parsed_args = parser.parse_args()
    
    # Run the generator
    use_cache = not parsed_args.no_cache
    exit_code = asyncio.run(run_generator(
        repo_url=parsed_args.repo_url,
        local_dir=parsed_args.local_dir,
        project_name=parsed_args.project_name,
        github_token=parsed_args.github_token,
        language=parsed_args.language,
        output_dir=parsed_args.output_dir,
        max_abstractions=parsed_args.max_abstractions,
        use_cache=use_cache
    ))
    
    if len(args) == 0:  # Only exit if called from command line
        sys.exit(exit_code)
    return exit_code


if __name__ == "__main__":
    main()
