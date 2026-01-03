from pathlib import Path
from logparser.Drain import LogParser
import pandas as pd

LOG_FORMAT = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
SIMILARITY_THRESHOLD = 0.5
TREE_DEPTH = 4

REGEX_PATTERNS = [
    r'core\.\d+', 
    r'(?<=r)\d{1,2}',                        # Register numbers
    r'(0x)?[0-9a-fA-F]{8}',                  # Hex addresses
    r'\d+\.\d+\.\d+\.\d+',                   # IP addresses
    r'\d+',                                  # Integers
    r'[a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+',  # Email-like patterns
    r'/[a-zA-Z0-9_/\-\.]+',                  # File paths
]

# def clean_log_file(input_path, output_path, max_lines=0):
#     # Clean log file - keep only safe ASCII characters to avoid parsing issues
#     print(f"Cleaning log file: {input_path}")
    
#     with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
#         with open(output_path, 'w', encoding='utf-8') as outfile:
#             for line in infile:
#                 cleaned = ''.join(c for c in line if c.isalnum() or c in ' .-_:/<>|[]()=+*\n\t')
#                 outfile.write(cleaned)
    
#     print(f"Cleaned log saved to: {output_path}")

def _clean_log_file(input_path, output_path, max_lines=None):
    print(f"Cleaning log file: {input_path}")
    if max_lines:
        print(f"Truncating input to first {max_lines} lines.")

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, line in enumerate(infile):
                # Stop if we have reached the limit
                if max_lines is not None and i >= max_lines:
                    break
                
                cleaned = ''.join(c for c in line if c.isalnum() or c in ' .-_:/<>|[]()=+*\n\t')
                outfile.write(cleaned)
    
    print(f"Cleaned log saved to: {output_path}")

def _count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)

def parse_logs(input_dir: str, log_filename: str, output_dir: str, truncate_proportion: float = 1.0) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    log_stem = Path(log_filename).stem 
    
    original_file = input_path / log_filename
    temp_cleaned_filename = f"cleaned_{log_stem}"
    temp_cleaned_file = input_path / temp_cleaned_filename
    
    expected_output = output_path / f"cleaned_{log_stem}_structured.csv"
    
    if expected_output.exists():
        print(f"Skipping: Output file '{expected_output.name}' already exists.")
        return

    print(f"Processing {log_filename}...")
    
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        max_lines_to_process = None
        
        if 0 < truncate_proportion < 1.0:
            total_lines = _count_lines(original_file)
            max_lines_to_process = int(total_lines * truncate_proportion)
            print(f"Processing only: {max_lines_to_process}")

        _clean_log_file(str(original_file), str(temp_cleaned_file), max_lines_to_process)

        parser = LogParser(
            LOG_FORMAT, 
            indir=str(input_path), 
            outdir=str(output_path), 
            depth=TREE_DEPTH, 
            st=SIMILARITY_THRESHOLD, 
            rex=REGEX_PATTERNS
        )

        parser.parse(temp_cleaned_filename)
        print("Parsing complete. Fles generated in {output_dir}: cleaned_{log_stem}_structured.csv and cleaned_{log_stem}_templates.csv")

    except Exception as e:
        print(f"Error parsing {log_filename}: {e}")
        raise
        
    finally:
        temp_cleaned_file.unlink(missing_ok=True)
        if not temp_cleaned_file.exists():
             print(f"Cleaned up temporary file: {temp_cleaned_filename}")
