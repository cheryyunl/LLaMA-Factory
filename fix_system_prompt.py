import json
import os
import shutil

def fix_jsonl_data(input_file, in_place=False):
    """
    Fix the language issue in jsonl data by correcting 'understanding3D' in system messages.
    
    Args:
        input_file (str): Path to the input jsonl file
        in_place (bool): Whether to overwrite the input file
    """
    temp_file = input_file + '.temp'
    
    # Read the input file line by line and write to a temporary file
    with open(input_file, 'r', encoding='utf-8') as fin, open(temp_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            # Parse the JSON line
            data = json.loads(line.strip())
            
            # Fix the system message if it exists and contains 'understanding3D'
            if 'messages' in data:
                for i, msg in enumerate(data['messages']):
                    if msg['role'] == 'system' and 'understanding3D' in msg['content']:
                        # Fix the language issue by changing 'understanding3D' to 'understanding 3D'
                        data['messages'][i]['content'] = msg['content'].replace(
                            'understanding3D', 'understanding 3D')
            
            # Write the fixed data to the temporary file
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    if in_place:
        # Replace the original file with the fixed file
        shutil.move(temp_file, input_file)
        print(f"Fixed data written back to {input_file}")
    else:
        # Create a new file with '_fixed' suffix
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_fixed{ext}"
        shutil.move(temp_file, output_file)
        print(f"Fixed data written to {output_file}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix language issues in jsonl data')
    parser.add_argument('input_file', help='Path to the input jsonl file')
    parser.add_argument('--in-place', action='store_true', help='Overwrite the input file')
    
    args = parser.parse_args()
    
    fix_jsonl_data(args.input_file, args.in_place)