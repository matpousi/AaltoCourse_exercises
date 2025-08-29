#!/usr/bin/env python3
"""
Jupyter Notebook Cell Tag Filter

This script processes Jupyter notebooks by removing cells that contain specified tags.
It can handle individual files, wildcard patterns, or directories.

Usage:
    python notebook_filter.py <path> [--tag TAG] [--suffix SUFFIX] [--recursive]
    
Examples:
    python notebook_filter.py notebook.ipynb --tag "remove"
    python notebook_filter.py "*.ipynb" --tag "debug" --suffix "_clean"
    python notebook_filter.py ./notebooks/ --tag "private" --recursive
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any


def find_notebooks(path_pattern: str, recursive: bool = False) -> List[Path]:
    """
    Find all Jupyter notebook files based on the input pattern.
    
    Args:
        path_pattern: File path, wildcard pattern, or directory
        recursive: Whether to search directories recursively
        
    Returns:
        List of Path objects for found notebook files
    """
    notebooks = []
    path = Path(path_pattern)
    
    if path.is_file() and path.suffix == '.ipynb':
        # Single file
        notebooks.append(path)
    elif path.is_dir():
        # Directory
        if recursive:
            notebooks.extend(path.rglob('*.ipynb'))
        else:
            notebooks.extend(path.glob('*.ipynb'))
    else:
        # Wildcard pattern
        for notebook_path in glob.glob(path_pattern, recursive=recursive):
            notebook_file = Path(notebook_path)
            if notebook_file.is_file() and notebook_file.suffix == '.ipynb':
                notebooks.append(notebook_file)
    
    return notebooks


def load_notebook(notebook_path: Path) -> Dict[str, Any]:
    """
    Load a Jupyter notebook from file.
    
    Args:
        notebook_path: Path to the notebook file
        
    Returns:
        Dictionary containing notebook data
        
    Raises:
        FileNotFoundError: If notebook file doesn't exist
        json.JSONDecodeError: If notebook file is not valid JSON
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid notebook format in {notebook_path}: {e}")


def cell_has_tag(cell: Dict[str, Any], tag: str) -> bool:
    """
    Check if a cell has a specific tag.
    
    Args:
        cell: Cell dictionary from notebook
        tag: Tag to search for
        
    Returns:
        True if cell contains the tag, False otherwise
    """
    metadata = cell.get('metadata', {})
    tags = metadata.get('tags', [])
    return tag in tags


def filter_cells(notebook: Dict[str, Any], exclude_tag: str) -> Dict[str, Any]:
    """
    Remove cells with specified tag from notebook.
    
    Args:
        notebook: Notebook dictionary
        exclude_tag: Tag of cells to remove
        
    Returns:
        Modified notebook dictionary
    """
    filtered_notebook = notebook.copy()
    original_cells = notebook.get('cells', [])
    
    # Filter out cells with the specified tag
    filtered_cells = [
        cell for cell in original_cells 
        if not cell_has_tag(cell, exclude_tag)
    ]
    
    filtered_notebook['cells'] = filtered_cells
    
    print(f"  Removed {len(original_cells) - len(filtered_cells)} cells with tag '{exclude_tag}'")
    print(f"  Kept {len(filtered_cells)} cells")
    
    return filtered_notebook


def save_notebook(notebook: Dict[str, Any], output_path: Path) -> None:
    """
    Save a notebook to file.
    
    Args:
        notebook: Notebook dictionary to save
        output_path: Path where to save the notebook
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)


def generate_output_path(input_path: Path, suffix: str) -> Path:
    """
    Generate output path with suffix.
    
    Args:
        input_path: Original notebook path
        suffix: Suffix to add to filename
        
    Returns:
        New path with suffix
    """
    stem = input_path.stem
    parent = input_path.parent
    extension = input_path.suffix
    
    return parent / f"{stem}{suffix}{extension}"


def process_notebook(notebook_path: Path, exclude_tag: str, suffix: str) -> None:
    """
    Process a single notebook file.
    
    Args:
        notebook_path: Path to input notebook
        exclude_tag: Tag of cells to exclude
        suffix: Suffix for output filename
    """
    print(f"Processing: {notebook_path}")
    
    try:
        # Load notebook
        notebook = load_notebook(notebook_path)
        
        # Filter cells
        filtered_notebook = filter_cells(notebook, exclude_tag)
        
        # Generate output path
        output_path = generate_output_path(notebook_path, suffix)
        
        # Save filtered notebook
        save_notebook(filtered_notebook, output_path)
        
        print(f"  Saved to: {output_path}")
        print()
        
    except Exception as e:
        print(f"  Error processing {notebook_path}: {e}")
        print()


def main():
    """Main function to handle command line arguments and process notebooks."""
    parser = argparse.ArgumentParser(
        description="Remove cells with specified tags from Jupyter notebooks",
        epilog="""
Examples:
  %(prog)s notebook.ipynb --tag remove
  %(prog)s "*.ipynb" --tag debug --suffix _clean  
  %(prog)s ./notebooks/ --tag private --recursive
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'path',
        help='File path, wildcard pattern, or directory containing notebooks'
    )
    
    parser.add_argument(
        '--tag', '-t',
        default='remove',
        help='Tag of cells to exclude (default: "remove")'
    )
    
    parser.add_argument(
        '--suffix', '-s', 
        default='_filtered',
        help='Suffix to add to output filenames (default: "_filtered")'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search directories recursively'
    )
    
    args = parser.parse_args()
    
    # Find all notebook files
    notebooks = find_notebooks(args.path, args.recursive)
    
    if not notebooks:
        print(f"No Jupyter notebooks found matching: {args.path}")
        return
    
    print(f"Found {len(notebooks)} notebook(s) to process")
    print(f"Excluding cells with tag: '{args.tag}'")
    print(f"Output suffix: '{args.suffix}'")
    print("-" * 50)
    
    # Process each notebook
    for notebook_path in notebooks:
        process_notebook(notebook_path, args.tag, args.suffix)
    
    print(f"Processing complete! Processed {len(notebooks)} notebook(s).")


if __name__ == "__main__":
    main()