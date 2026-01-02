from pathlib import Path
from logparser.Drain import LogParser

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

def clean_log_file(input_path, output_path):
    # Clean log file - keep only safe ASCII characters to avoid parsing issues
    print(f"Cleaning log file: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                cleaned = ''.join(c for c in line if c.isalnum() or c in ' .-_:/<>|[]()=+*\n\t')
                outfile.write(cleaned)
    
    print(f"Cleaned log saved to: {output_path}")

def parse_logs(input_dir: str, log_filename: str, output_dir: str) -> None:    
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
        clean_log_file(str(original_file), str(temp_cleaned_file))

        parser = LogParser(
            LOG_FORMAT, 
            indir=str(input_path), 
            outdir=str(output_path), 
            depth=TREE_DEPTH, 
            st=SIMILARITY_THRESHOLD, 
            rex=REGEX_PATTERNS
        )

        parser.parse(temp_cleaned_filename)
        
    except Exception as e:
        print(f"Error parsing {log_filename}: {e}")
        raise
        
    finally:
        temp_cleaned_file.unlink(missing_ok=True)
        if not temp_cleaned_file.exists():
             print(f"Cleaned up temporary file: {temp_cleaned_filename}")
