import os
from pathlib import Path

def create_val_file():
    """
    Create val.txt by scanning all .txt files in the labels directory.
    Each .txt file's basename (without extension) is added to val.txt.
    """
    # Define paths
    labels_dir = Path(__file__).parent / 'labels'
    imagesets_dir = Path(__file__).parent / 'ImageSets'
    val_file = imagesets_dir / 'val.txt'
    
    # Create ImageSets directory if it doesn't exist
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in labels directory
    txt_files = sorted(labels_dir.glob('*.txt'))
    
    # Extract basenames (without .txt extension)
    basenames = [f.stem for f in txt_files]
    
    # Sort numerically (convert to int if possible, otherwise alphabetically)
    try:
        basenames.sort(key=lambda x: int(x))
    except ValueError:
        basenames.sort()
    
    # Truncate the val.txt file first (ensure clean slate)
    if val_file.exists():
        val_file.unlink()
    
    # Write to val.txt
    with open(val_file, 'w') as f:
        for basename in basenames:
            f.write(f"{basename}\n")
    
    print(f"Created {val_file} with {len(basenames)} entries")
    print(f"First few entries: {basenames[:5]}")
    print(f"Last few entries: {basenames[-5:]}")

if __name__ == '__main__':
    create_val_file()
