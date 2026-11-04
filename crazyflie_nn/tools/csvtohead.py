import argparse
import csv
import os
from pathlib import Path

def convert_folder_to_single_header(input_folder, output_filename):
    input_path = Path(input_folder)
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a valid directory.")
        return

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as h:
        h.write("#ifndef MATRICES_H\n")
        h.write("#define MATRICES_H\n\n")

        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in the directory: {input_folder}")
            return

        for csv_file in csv_files:
            var_name = csv_file.stem.replace(" ", "_").replace("-", "_") # Sanitize variable name

            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                # Filter out empty rows and convert to flat list of strings
                data = [row for row in reader if any(field.strip() for field in row)]

                if not data:
                    continue
                rows = len(data)
                cols = len(data[0])
                flat_data = [val.strip() for row in data for val in row]

            
            if rows == 1 and cols > 1:
                rows, cols = cols, rows

            # 3. Write the static raw data array
            h.write(f"/* Data for {csv_file.name} */\n")
            h.write(f"static float {var_name}_raw[] = {{\n")

            if cols == 1:
                for val in flat_data:
                    h.write(f"    {val},\n")
            else:
                for i in range(0, len(flat_data), 8):
                    chunk = ", ".join(flat_data[i:i+8])
                    h.write(f"    {chunk},\n")

            h.write("};\n\n")

            h.write(f"static const Matrix {var_name} = {{\n")
            h.write(f"    .rows = {rows},\n")
            h.write(f"    .cols = {cols},\n")
            h.write(f"    .data = {var_name}_raw\n")
            h.write("};\n\n")

        h.write("#endif // MATRICES_H\n")

    print(f"Done! Created '{output_path}' with {len(csv_files)} matrices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of CSV files into a single C header file with matrix data.")

    parser.add_argument("input_folder", type=str,
                        help="Path to the folder containing the CSV files.")

    parser.add_argument("-o", "--output", type=str, default="matrices.h",
                        help="Path to the output header file (default: matrices.h).")

    args = parser.parse_args()

    convert_folder_to_single_header(args.input_folder, args.output)
