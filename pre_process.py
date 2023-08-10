import discorpy
import discorpy.losa.loadersaver as io
import discorpy.prep.preprocessing as prep
import discorpy.prep.linepattern as lprep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post

import os

import numpy as np

def calculate_coefficients_radial_distortion(file_path: str, output_base: str, num_coef: int = 5):
    """
    Objective: Calculate the coefficients for radial distortion correction and save the results.

    Input:
    - file_path (str): The file path to the input image.
    - output_base (str): The base directory for saving output files.
    - num_coef (int, optional): Number of polynomial coefficients. Default is 5.

    Output: None (Results are saved in files).
    """
    mat0 = io.load_image(file_path) # Load image
    (height, width) = mat0.shape

    # Convert the chessboard image to a line-pattern image
    mat1 = lprep.convert_chessboard_to_linepattern(mat0)

    # Calculate slope and distance between lines
    slope_hor, dist_hor = lprep.calc_slope_distance_hor_lines(mat1, radius=15, sensitive=0.5)
    slope_ver, dist_ver = lprep.calc_slope_distance_ver_lines(mat1, radius=15, sensitive=0.5)

    # Extract reference-points
    list_points_hor_lines = lprep.get_cross_points_hor_lines(mat1, slope_ver, dist_ver,
                                                            ratio=0.3, norm=True, offset=450,
                                                            bgr="bright", radius=15,
                                                            sensitive=0.5, denoise=True,
                                                            subpixel=True)
    list_points_ver_lines = lprep.get_cross_points_ver_lines(mat1, slope_hor, dist_hor,
                                                            ratio=0.3, norm=True, offset=150,
                                                            bgr="bright", radius=15,
                                                            sensitive=0.5, denoise=True,
                                                            subpixel=True)
    if len(list_points_hor_lines) == 0 or len(list_points_ver_lines) == 0:
        raise ValueError("No reference-points detected !!! Please adjust parameters !!!")

    # Group points into lines
    list_hor_lines = prep.group_dots_hor_lines(list_points_hor_lines, slope_hor, dist_hor,
                                            ratio=0.1, num_dot_miss=2, accepted_ratio=0.8)
    list_ver_lines = prep.group_dots_ver_lines(list_points_ver_lines, slope_ver, dist_ver,
                                            ratio=0.1, num_dot_miss=2, accepted_ratio=0.8)

    # Remove residual dots
    list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 2.0)
    list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 2.0)

    # Regenerate grid points after correcting the perspective effect.
    list_hor_lines, list_ver_lines = proc.regenerate_grid_points_parabola(
    list_hor_lines, list_ver_lines, perspective=True)

    # Calculate parameters of the radial correction model
    (xcenter, ycenter) = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
    list_fact = proc.calc_coef_backward(list_hor_lines, list_ver_lines,
                                        xcenter, ycenter, num_coef)
    print('OK')
    io.save_metadata_txt(output_base + "/coefficients_radial_distortion.txt",
                        xcenter, ycenter, list_fact)


def process_image_optical_correction(coefficients_file: str, input_dir: str, output_dir: str):
    """
    Apply optical correction to all images in the input directory and save the results in the output directory.

    Inputs:
    - coefficients_file (str): The path to the text file containing the coefficients.
    - input_dir (str): The path to the directory containing different images.
    - output_dir (str): The path to the output directory for saving the corrected images.
    """
    # Load coefficients from the provided file
    (xcenter, ycenter, list_fact) = io.load_metadata_txt(coefficients_file)

    # Check if the output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        # Check if the file is an image (you can update the file extension to match your image format)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load the image and correct it
            img = io.load_image(os.path.join(input_dir, file_name), average=False)
            img_corrected = np.copy(img)
            for i in range(img.shape[-1]):
                img_corrected[:, :, i] = post.unwarp_image_backward(img[:, :, i], xcenter, ycenter, list_fact)

            # Save the corrected image to the output directory
            output_file = os.path.join(output_dir, file_name)
            io.save_image(output_file, img_corrected)
