#!/bin/bash

# Assign input arguments to variables
script_file="/home/mcaroulle/Pheno_Leaf/PL_Final/pre_process.py"  # Replace this path with the correct path to your Python script
input_dir="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_INPUT"

### Optical correction distortion

# Extract input arguments

#Path of the chessboard calibration image
file_path="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_INPUT/Real-chessboard.JPG" 
#Path to store the results 
output_base="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_OUTPUT"

# Check if the file exists
if [ ! -f "$file_path" ]; then
    echo "Error: Input file not found: $file_path"
    exit 1
fi

# Check if the output directory exists, if not create it
if [ ! -d "$output_base" ]; then
    mkdir -p "$output_base"
fi

# Call the Python script with the provided input arguments
python -c "from pre_process import calculate_coefficients_radial_distortion; calculate_coefficients_radial_distortion('$file_path', '$output_base')"


coefficients_file="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_OUTPUT/coefficients_radial_distortion.txt"
input_dir="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_INPUT"
output_dir="/home/mcaroulle/Pheno_Leaf/PL_Final/RAW_OUTPUT"

# Check if the coefficients file exists
if [ ! -f "$coefficients_file" ]; then
    echo "Error: Coefficients file not found: $coefficients_file"
    exit 1
fi

# Call the Python script with the provided input arguments
python -c "from pre_process import process_image_optical_correction; process_image_optical_correction('$coefficients_file','$input_dir','$output_dir')"


