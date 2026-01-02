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

# def parse_logs(file_path, log_file_name, output_dir):
#     log_file = log_file_name.split(".")[0]

#     # Check if output file already exists
#     expected_output = os.path.join(output_dir, f'cleaned_{log_file}_structured.csv')
#     if os.path.exists(expected_output):
#         print(f"File '{expected_output}' already exists. Skipping parsing.")
#         return
    
#     print(f"Parsing {log_file_name}...")

#     # Clean the log file first
#     original_log = os.path.join(file_path, log_file_name)
#     cleaned_log = os.path.join(file_path, f'cleaned_{log_file_name}')
    
#     try:
#         clean_log_file(original_log, cleaned_log)

#         log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
#         similarity_threshold = 0.5
#         depth = 4

#         regex = [
#             r'core\.\d+', 
#             r'(?<=r)\d{1,2}',                        # Register numbers
#             r'(0x)?[0-9a-fA-F]{8}',                  # Hex addresses
#             r'\d+\.\d+\.\d+\.\d+',                   # IP addresses
#             r'\d+',                                  # Integers
#             r'[a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-\.]+',  # Email-like patterns
#             r'/[a-zA-Z0-9_/\-\.]+',                  # File paths
#         ]

#         os.makedirs(output_dir, exist_ok=True)

#         parser = LogParser(log_format, indir=file_path, outdir=output_dir, depth=depth, st=similarity_threshold, rex=regex)
#         parser.parse(f'cleaned_{log_file}')
    
#     finally:
#          if os.path.exists(cleaned_log):
#             os.remove(cleaned_log)
#             print(f"Removed temporary file: {cleaned_log}")
