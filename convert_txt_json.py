import json
import logging
from pathlib import Path

def convert_to_json(input_file, output_file):
    data = {}
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    var_name = parts[0]
                    try:
                        min_val = float(parts[1]) 
                        max_val = float(parts[2])
                    except ValueError as e:
                        logging.warning(f"Skipping line, could not convert values: {line}")
                        continue
                        
                    var_name = var_name.replace('_float', '')
                    
                    data[var_name] = {
                        "min": min_val,
                        "max": max_val
                    }

        # Write JSON file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
            
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Convert scaler file to JSON format")
    #parser.add_argument("-m", "--mass_diff", help="mass difference", type=str, choices=["small", "large"], required=True)
    parser.add_argument("-f", "--flavor", choices=["SF0b", "SF1b", "DF0b", "DF1b"], default="SF0b")
    parser.add_argument("-e", "--event_type", help="odd or even", type=str, choices=['odd','even'], required=True)
    #parser.add_argument("-n", "--nbjets", type=int, help="number of b-jets", choices=[0,1,2], required=True)
    
    args = parser.parse_args()
    # scaler_largeDM_bjet0_even.txt
    #input_file = Path(f"scaler_{args.mass_diff}DM_bjet{args.nbjets}_{args.event_type}.txt")
    input_file = Path(f"scaler_{args.flavor}_{args.event_type}.txt")
    output_file = input_file.with_suffix('.json')
    
    convert_to_json(input_file, output_file)