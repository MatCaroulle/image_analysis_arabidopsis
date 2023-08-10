#!/bin/bash

# Assign input arguments to variables
script_file="/home/mcaroulle/Pheno_Leaf/PL_Final/PL_code.py"  # Replace this path with the correct path to your Python script
input_dir="/home/mcaroulle/Pheno_Leaf/PL_Final/SEGMENTED_OUTPUT"


# Check if the script file and input directory exist
if [ ! -f "$script_file" ]; then
    echo "Error: Script file '$script_file' not found."
    exit 1
fi

if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' not found."
    exit 1
fi

# Create an empty CSV file to store the results
output_csv="/home/mcaroulle/Pheno_Leaf/PL_Final/results.csv"
echo "Filename,Max_Distance,Area,Convex area, Equivalent diameter area, Solidity, Eccentricity" > "$output_csv"

# Loop over the files in the input directory
find "$input_dir" -type f -print0 | while IFS= read -r -d '' file; do
    # Execute the Python script with the find_diameter() function on each file
    max_distance=$(python -c "import sys; sys.path.append('/home/mcaroulle/Pheno_Leaf/PL_Final/'); from PL_code import find_diameter; print(find_diameter('$file'))")
    
    TUPLE=$(python -c "import sys; sys.path.append('/home/mcaroulle/Pheno_Leaf/PL_Final/'); from PL_code import extract_regionprops_data; print(extract_regionprops_data('$file'))")
    TUPLE="${TUPLE#*(}"
    TUPLE="${TUPLE%)*}"
    
    Area=$(echo "$TUPLE" | python -c "import sys; data=sys.stdin.read(); print(data.split(',')[0].strip())")
    
    Convex_Area=$(echo "$TUPLE" | python -c "import sys; data=sys.stdin.read(); print(data.split(',')[1].strip())")
    
    Equuivalent_diameter_area=$(echo "$TUPLE" | python -c "import sys; data=sys.stdin.read(); print(data.split(',')[2].strip())")
    Solidity=$(echo "$TUPLE" | python -c "import sys; data=sys.stdin.read(); print(data.split(',')[3].strip())")
    Eccentricity=$(echo "$TUPLE" | python -c "import sys; data=sys.stdin.read(); print(data.split(',')[4].strip())")





    filename=$(basename "$file")
    k=$(echo "$filename" | grep -oP '_plant_\K\d+')
    #breaky
    # Append k and max_distance to the CSV file
    echo "$k,$max_distance,$Area,$Convex_Area,$Equuivalent_diameter_area,$Solidity,$Eccentricity" >> "$output_csv"
done

echo "Processing complete. Results are stored in 'results.csv'."