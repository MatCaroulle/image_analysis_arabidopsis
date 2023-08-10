import cv2

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from skimage.morphology import disk
from skimage import graph, measure
from skimage.morphology import skeletonize
from skimage import feature
from skimage import io, feature
from skimage import  color
from skimage.color import label2rgb

import scipy.ndimage as nd
from scipy.ndimage import distance_transform_edt

import os


########################################################################################################################################################################################################################## Technical functions 
def show_image(image, title):
    """
    Function to display the image.
    
    Objective: Display an image using the matplotlib library.
    
    Input:
        image (numpy.ndarray) - The image as a NumPy array.
        title (str) - The title of the image to display.
    
    Output:
        None (displays the image in a matplotlib window).
    """
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def convert_mask_to_image(mask):
    """
    Objective:
    The objective of this function is to convert a binary mask into a corresponding binary image.

    Input variables and their types:
    - mask (numpy array): The binary mask to be converted. This variable should be a NumPy array representing a binary image,
                          where pixel values are either 0 or 1.

    Output variables and their types:
    - image (numpy array): The resulting binary image after converting the input mask. The output is also a NumPy array representing
                           a binary image, where pixel values are either 0 (black) or 255 (white).
    """

    # Create an empty image with the same dimensions as the mask
    image = np.zeros_like(mask)

    # Apply the mask to the image by setting white pixels
    image[mask > 0] = 255

    # Return the resulting binary image
    return image

def image_to_mask(image_path, threshold=240):
    """
    Convert an 8-bit grayscale image to a binary mask.

    Parameters:
        image_path (str): The path to the input 8-bit grayscale image.
        threshold (float, optional): Threshold value for binarizing the image. Default is 0.5.

    Returns:
        np.ndarray: The binary mask as a NumPy array.
    """
    # Load the image
    image = io.imread(image_path)

    # If the image is color, convert it to grayscale
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Binarize the grayscale image using the threshold value
    mask = (image > threshold).astype(np.uint8) * 255

    return mask

def extract_filename_from_path(file_path):
    """
    Extract the filename (without extension) from a file path.

    Parameters:
        file_path (str): The path to the file.

    Returns:
        str: The extracted filename without extension.
    """
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

##########################################################################################################################################################################################################################Image segmentation 

############################################################# Color segmentation
def color_segmentation(image_path):
    """
    Objective:
    This function performs color segmentation on an input image and generates a mask based on specified color ranges.

    Input variables:
    - image_path (str): The path to the input image.

    Output variables:
    - mask (numpy array): The binary mask obtained after color segmentation.
    """

    # Read the input image
    img = io.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define the color range for segmentation (example: green color range)
    lower_range = (30, 20, 150)
    upper_range = (70, 255, 255)

    # Create a binary mask based on the color range
    mask = cv2.inRange(hsv, lower_range, upper_range)

    return mask



############################################################# Measurements on mask to help processing future steps
def mask_measurements(mask):
    """
    Objective:
    The objective of this function is to extract measurements from an input binary mask and create a filtered dataframe.

    Input variables and their types:
    - mask (numpy array): The binary mask from which measurements are to be extracted. This variable should be a NumPy array
                          representing a binary image, where pixel values are either 0 or 1.

    Output variables and their types:
    - df_filtered (pandas DataFrame): The resulting filtered dataframe containing measurements from the input mask.
    """

    # Apply binary closing and remove border artifacts from the mask
    closed_mask = mask_closing(mask)

    # Label connected components in the binary mask
    label_image = measure.label(closed_mask)

    # Extract region properties (label, area, coordinates) from the labeled mask
    props = measure.regionprops_table(label_image, properties=['label', 'area', 'coords'])

    # Create a dataframe from the region properties
    df_filtered = pd.DataFrame(props)

    return df_filtered


def mask_closing(mask):
    """
    Objective:
    This function performs binary closing on an input mask to fill small holes and smooth object edges.

    Input variables:
    - mask (numpy array): The binary mask to be filtered. It should be a NumPy array representing a binary image,
                          where pixel values are either 0 or 1.

    Output variables:
    - filtered_mask (numpy array): The binary mask after applying binary closing. It is also a NumPy array
                                   representing a binary image.
    """

    # Perform binary closing on the input mask using a 7x7 square structuring element
    closed_mask = nd.binary_closing(mask, np.ones((7, 7)))


    return closed_mask


############################################################# Detecting leaves and locate them

def peak_identification(mask):
    """
    Objective:
    The objective of this function is to identify peak points in an input binary mask.

    Input variables and their types:
    - mask (numpy array): The binary mask for which peak points are to be identified. This variable should be a NumPy array representing
                          a binary image, where pixel values are either 0 or 1.

    Output variables and their types:
    - peak_idx (numpy array): An array containing the coordinates of peak points found in the input mask. Each row represents a peak point,
                              and the columns contain the (row, column) coordinates of each point.
    """

    # Convert the input binary mask to a binary image
    image = convert_mask_to_image(mask)

    # Detect edges using Canny edge detection with a sigma of 10
    edges = feature.canny(image, sigma=10)

    # Calculate the distance transform of the complement of edges
    dt = distance_transform_edt(~edges)

    # Identify local maxima in the distance transform with a minimum distance of 5 pixels
    peak_idx = feature.peak_local_max(dt, min_distance=5)

    return peak_idx




def label_coordinates(mask, df_filtered):
    """
    Objective:
    The objective of this function is to label coordinates of peak points in an input binary mask using a given filtered dataframe.

    Input variables and their types:
    - mask (numpy array): The binary mask for which peak points are to be identified. This variable should be a NumPy array representing
                          a binary image, where pixel values are either 0 or 1.
    - df_filtered (pandas DataFrame): The filtered dataframe containing labels and corresponding coordinates.

    Output variables and their types:
    - df_peak_label (pandas DataFrame): The resulting dataframe after labeling the peak points. It contains the filtered dataframe
                                        columns along with the 'x_coord' and 'y_coord' columns representing the peak coordinates.
    """

    # Peak dataframe
    peak_idx = peak_identification(mask)
    df_peak = pd.DataFrame(peak_idx)
    df_peak.columns = ['x_coord', 'y_coord']
    df_peak['index_of_label_in_df_coords'] = ''
    df_peak['label'] = ''

    df_coords = df_filtered[['label', 'coords']]

    for k in range(len(df_coords)): 
        data_coords = df_coords['coords'][k]  # array
        coords = data_coords.tolist()  # list of list
        x_coords = [sous_liste[0] for sous_liste in coords]  # list
        y_coords = [sous_liste[1] for sous_liste in coords]  # list 

        # Iterate over the rows of the peak dataframe
        for index, row in df_peak.iterrows():
            if row['x_coord'] in x_coords and row['y_coord'] in y_coords:
                df_peak.at[index, 'index_of_label_in_df_coords'] = k

    # Cleaning the obtained dataframe
    df_peak = df_peak[df_peak['index_of_label_in_df_coords'] != '']

    # Find duplicate values in column 'index_of_label_in_df_coords'
    valeurs_en_double = df_peak[df_peak.duplicated(subset='index_of_label_in_df_coords', keep=False)]['index_of_label_in_df_coords'].unique()

    # Remove duplicate rows, keeping the first occurrence
    df_peak = df_peak.drop_duplicates(subset='index_of_label_in_df_coords', keep='first')

    for k in df_peak['index_of_label_in_df_coords']: 
        # Extract the element from column 1 based on the index
        element = df_coords.loc[k, 'label']

        # Find the index of the row containing the element in column 3
        index = df_peak.loc[df_peak['index_of_label_in_df_coords'] == k].index[0]

        # Add the element to the specified row in column 3
        df_peak.at[index, 'label'] = element

    # Merge the filtered dataframe and the peak dataframe based on the 'label' column
    df_peak_label = pd.merge(df_filtered, df_peak, on='label', how='inner')
    df_peak_label = df_peak_label.rename(columns={'x_coord': 'y_coord', 'y_coord': 'x_coord'})

    return df_peak_label



############################################################# Tray slicing



def tray_slicing(image_path,r,c):
    """
    Objective:
    The objective of this function is to slice a tray image into individual pots and create two dataframes containing pot boundaries.
    
    Input variables and their types:
    - image_path (str): The file path of the tray image to be sliced.
    
    Output variables and their types:
    - df_tray_min (pandas DataFrame): The dataframe containing the minimum boundaries of individual pots.
    - df_tray_max (pandas DataFrame): The dataframe containing the maximum boundaries of individual pots.
    - r (int): The number of rows of pots in the tray.
    - c (int): The number of columns of pots in the tray.
    """

    # Load the tray image
    image = io.imread(image_path)

    if image.ndim == 2:
        # Grayscale image: has no channels
        height, width = image.shape
        channels = 1  # For grayscale images, set channels to 1
    elif image.ndim == 3:
        # Color image: has multiple channels
        height, width, channels = image.shape
    else:
        raise ValueError("Unexpected number of dimensions in the image.")


    # Calculate the width and height of each pot in the tray
    x_pot = width / c
    y_pot = height / r

    # Create data for slicing the tray into pots
    x_min = []
    x_max = []
    X = []

    # Slicing the tray horizontally (along x-axis)
    for k in range(r):
        x = 0
        for k in range(c + 1):
            X.append(int(x))
            x += x_pot
        x_min += X[0:c]
        x_max += X[1:c + 1]
        if x_max[-1] != width:
            x_max[-1] = width

    x_min.sort()
    x_max.sort()

    y_min = []
    y_max = []
    Y = []

    # Slicing the tray vertically (along y-axis)
    for k in range(c):
        y = 0
        for k in range(r + 1):
            Y.append(int(y))
            y += y_pot
        Y.sort()
        y_min += Y[0:r]
        y_max += Y[1:r + 1]
        Y = []
        if y_max[-1] != height:
            y_max[-1] = height

    # Create dataframe df_tray_max with pot boundaries
    data_tray_max = {'Pot_position': [i for i in range(1, 36)],
                     'xmin': [k for k in x_min],
                     'xmax': [k for k in x_max],
                     'ymin': [k for k in y_min],
                     'ymax': [k for k in y_max]}
    
    df_tray_max = pd.DataFrame(data_tray_max)
    
    # Create dataframe df_tray_min as a copy of df_tray_max and adjust boundaries to create a rejected zone
    df_tray_min = df_tray_max.copy()
    x_R_pot = int((width / 7) / 20)
    y_R_pot = int((height / 5) / 20)
    df_tray_min['xmin'] = df_tray_min['xmin'] + x_R_pot
    df_tray_min['xmax'] = df_tray_min['xmax'] - x_R_pot
    df_tray_min['ymin'] = df_tray_min['ymin'] + y_R_pot
    df_tray_min['ymax'] = df_tray_min['ymax'] - y_R_pot
    
    return df_tray_min, df_tray_max


############################################################# Detecting plants and assign them to a pot


def assign_pot_to_peak(df_peak_label, df_tray_min):
    """
    1- Function Objective:
    Assigns a 'pot' value to each row in the 'df_peak_label' DataFrame based on its coordinates and matching 'pot'
    from the 'df_tray_min' DataFrame.

    2- Input Variables and their types:
    - df_peak_label (pd.DataFrame): DataFrame containing peak labels with columns 'x_coord', 'y_coord', and 'pot'.
    - df_tray_min (pd.DataFrame): DataFrame containing tray coordinates with columns 'xmin', 'xmax', 'ymin', 'ymax', and 'Pot_position'.

    3- Output Variables and their types:
    - df_peak_label (pd.DataFrame): Updated DataFrame 'df_peak_label' with 'pot' values assigned based on matching coordinates.
    """

    # Add an empty column named 'pot' to the DataFrame df_peak_label
    df_peak_label['pot'] = ''

    # Iterate through df_tray_min and assign the corresponding 'pot' to each label
    indices_presents = []

    for k in range(len(df_tray_min)):
        # Filter the points in df_peak_label based on coordinates
        df_subset_peak_label_min = df_peak_label[
            (df_peak_label['x_coord'] >= df_tray_min['xmin'][k]) &
            (df_peak_label['x_coord'] <= df_tray_min['xmax'][k]) &
            (df_peak_label['y_coord'] >= df_tray_min['ymin'][k]) &
            (df_peak_label['y_coord'] <= df_tray_min['ymax'][k])
        ]

        pot = df_tray_min.loc[k, 'Pot_position']

        index = df_subset_peak_label_min.index.tolist()

        if len(index) > 0:
            for k in index:
                indices_presents.append(k)

        for k in index:
            df_peak_label.loc[k, 'pot'] = pot

    # Remove all labels that are not associated with a 'pot'
    indices_absents = [k for k in df_peak_label.index.tolist() if k not in indices_presents]
    df_peak_label = df_peak_label.drop(indices_absents)

    # Sort the DataFrame based on the 'pot' column in ascending order
    df_peak_label = df_peak_label.sort_values(by='pot', ascending=True)

    return df_peak_label


def merge_to_plant(label_image, df_peak_label):
    """
    1- Function Objective:
    Merge regions from 'label_image' based on 'df_peak_label' and return a list of fused masks.

    2- Input Variables and their types:
    - label_image (np.ndarray): The label image containing regions/segments.
    - df_peak_label (pd.DataFrame): DataFrame containing peak labels with columns 'label' and 'pot'.

    3- Output Variables and their types:
    - masquefusione_list (list of np.ndarray): List containing fused masks for each plant.
    """

    masquefusione_list = []  # List to store the fused masks at each iteration of 'i'

    for i in range(1, 36):
        masquefusione = np.zeros_like(label_image, dtype=np.uint8)  # Default value, empty array
        Labels = df_peak_label.loc[df_peak_label['pot'] == i, 'label'].tolist()

        for k in range(len(Labels)):
            globals()[f'masque_label{k + 1}_{i}'] = np.where(label_image == Labels[k], 255, 0).astype(np.uint8)
            if k == 0 and len(Labels) > 1:
                pass
            elif k == 0 and len(Labels) == 1:
                masquefusione = globals()[f'masque_label{k + 1}_{i}']
            elif k == 1:
                masquefusione = np.logical_or(globals()[f'masque_label{k}_{i}'], globals()[f'masque_label{k + 1}_{i}'])
            else:
                masquefusione = np.logical_or(masquefusione, globals()[f'masque_label{k + 1}_{i}'])

        masquefusione_list.append(masquefusione.copy())  # Store a copy of 'masquefusione' at each iteration

    return masquefusione_list




############################################################# Extract plant images from the tray 

def split_masks_with_bbox(masks, output_dir, mask_path):
    """
    1- Function Objective:
    Split the input masks into smaller sub-masks using bounding boxes with a 5% increase in size while keeping the center fixed,
    and save the resulting sub-masks to the specified output directory.

    2- Input Variables and their types:
    - masks (list of np.ndarray): List containing input masks represented as NumPy arrays.
    - output_dir (str): The directory path where the resulting sub-masks will be saved.
    - mask_path (str): The file path of the mask image that has been analysed previously.

    3- Output Variables and their types:
    - None: The function does not return any values. It saves the sub-masks as image files to the specified 'output_dir'.
    """
    input_file_name = extract_filename_from_path(mask_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, mask in enumerate(masks):
        # Convert the mask to np.uint8 data type
        mask = mask.astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            continue

        # Compute the bounding box of all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Calculate the new dimensions of the bounding box with a 5% increase
        new_w = int(w * 1.05)
        new_h = int(h * 1.05)

        # Calculate the new top-left corner (x, y) to keep the center of the bounding box the same
        new_x = int(x - (new_w - w) / 2)
        new_y = int(y - (new_h - h) / 2)

        # Ensure the new bounding box is within the image bounds
        new_x = max(new_x, 0)
        new_y = max(new_y, 0)
        new_x_end = min(new_x + new_w, mask.shape[1])
        new_y_end = min(new_y + new_h, mask.shape[0])

        # Create a blank mask of the same size as the input mask
        sub_mask = np.zeros_like(mask)

        # Draw all contours on the blank mask
        cv2.drawContours(sub_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Crop the sub-mask using the new bounding box
        sub_mask = sub_mask[new_y:new_y_end, new_x:new_x_end]

        # Save the sub-mask to the output directory
        sub_mask_path = os.path.join(output_dir, f"sub_mask_{input_file_name}_plant_{i+1}.png")
        cv2.imwrite(sub_mask_path, sub_mask)





############################################################# Rosette center

def has_multiple_objects(mask):
    """
    Function to check if a mask contains multiple objects.
    
    Objective: Check if a binary mask contains multiple distinct objects using contour detection.
    
    Input:
        mask (numpy.ndarray) - The binary mask as a NumPy array.
    
    Output:
        bool - True if the mask contains multiple objects, False otherwise.
    """
    # Find contours in the binary mask (working on a copy)
    mask_copy=mask.copy()
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if there is more than one contour (object)
    return len(contours) > 1



def process_mask_dilation(mask_path):
    """
    Function to process dilation the mask and obtain the final mask and the value of n.
    
    Objective: Process the mask by performing repeated dilations until it contains only one object. (e.g. analogy with bacterial colony growth with contact inhibition growth)
    
    Input:
        mask_path (str) - The path to the mask file to process.
    
    Output:
        final_mask (numpy.ndarray) - The final binary mask containing a single object as a NumPy array.
        n_value (int) - The value of n indicating the number of dilations performed to obtain the final mask.
    """
    # Load the mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    # Check if the mask contains multiple objects
    n = 0
    while has_multiple_objects(mask):
        n += 1

        # Perform dilation with a footprint = disk(n) (working on a copy)
        dilated_mask = np.copy(mask)
        footprint = disk(n)
        dilated_mask = cv2.dilate(dilated_mask, footprint)

        # Update the mask for the next iteration
        mask = dilated_mask

    # Return the final mask and the value of n
    return mask, n


def Rosette_center_coord(mask, n):
    """
    Function to process a mask and return the coordinates based on the value of n.
    
    Objective: Process the binary mask and compute the coordinates of the graph center or centroid depending on the value of n.
    
    Input:
        mask (numpy.ndarray) - The binary mask as a NumPy array.
        n (int) - The integer value to determine whether to compute the graph center or the centroid.
    
    Output:
        tuple (float, float) - The coordinates (x, y) of either the graph center or the centroid.
    """
    # Convert the mask to binary
    binary_mask = mask > 0

    # Perform skeletonization
    skeleton = skeletonize(binary_mask)

    if n < 5:
        # Get the graph center coordinates for n < 5
        g, nodes = graph.pixel_graph(skeleton, connectivity=2)
        px, distances = graph.central_pixel(g, nodes=nodes, shape=skeleton.shape, partition_size=100)

        # Return the coordinates of the graph center
        return px[1], px[0]
    else:
        # Get the centroid coordinates for n >= 5
        centroid = measure.centroid(skeleton)

        # Return the coordinates of the centroid
        return centroid[1], centroid[0]
    

def find_farthest_distance(mask_path, x, y):
    """
    Function to find the farthest distance from a given point (x, y) to any point in the contours of a mask.
    
    Objective: Find the largest distance from a given point (x, y) to any point in the contours of a binary mask.
    
    Input:
        mask_path (str) - The path to the mask file.
        x (int) - The x coordinate of the point to calculate the distance from.
        y (int) - The y coordinate of the point to calculate the distance from.
    
    Output:
        float - The largest distance from the given point to any point in the contours.
    """
    # Load the image and apply edge detection
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    edges = feature.canny(image, sigma=0.6)

    # Find contours in the image
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the coordinates of the farthest point
    max_distance = 0
    farthest_point = None

    # Find the farthest point from the given coordinates (x, y) among all the contours
    for contour in contours:
        for point in contour:
            distance = np.linalg.norm(np.subtract([x, y], point[0]))
            if distance > max_distance:
                max_distance = distance
                farthest_point = point[0]

    # Return the largest distance
    return max_distance

def find_diameter(mask_path):
    """
    # Objective: Calculate the maximum distance from a given mask file to any point in its contours.
    
    # Input:
        mask_path (str) - The path to the mask file to process.
    
    # Output:
        max_distance (float) - The maximum distance from a given point to any point in the contours of the mask.
    """
    # Process the mask to obtain the final mask and the value of n
    final_mask, n_value = process_mask_dilation(mask_path)

    if n_value < 5:
        # Get the coordinates of the graph center for n < 5
        x, y = Rosette_center_coord(final_mask, n_value)
    else:
        # Get the coordinates of the centroid for n >= 5
        centroid = measure.centroid(final_mask)
        x, y = centroid[1], centroid[0]

    # Find the farthest distance from the given coordinates to any point in the contours of the mask
    max_distance = find_farthest_distance(mask_path, x, y)

    # Return the result (max_distance)
    return max_distance

def extract_regionprops_data(image):

    
    properties = ['area', 'area_convex', 'equivalent_diameter_area', 'solidity', 'eccentricity']
    props = measure.regionprops_table(image, properties=properties)
    
    data_tuple_list = []
    num_regions = len(props['area'])  # Nombre de régions trouvées
    
    for i in range(num_regions):
        data_tuple = tuple(props[property_name][i] for property_name in properties)
        data_tuple_list.append(data_tuple)
    
    return tuple(data_tuple_list[0])


def extract_regionprops_data(image_path):
    """
    # Objective: Calculate the maximum distance from a given mask file to any point in its contours.
    
    # Input:
        mask_path (str) - The path to the mask file to process.
    
    # Output:
        max_distance (float) - The maximum distance from a given point to any point in the contours of the mask.
    """
    
    image=io.imread(image_path)
    properties = ['area', 'area_convex', 'equivalent_diameter_area', 'solidity', 'eccentricity']
    props = measure.regionprops_table(image, properties=properties)
    
    data_tuple_list = []
    num_regions = len(props['area'])  # Nombre de régions trouvées
    
    for i in range(num_regions):
        data_tuple = tuple(props[property_name][i] for property_name in properties)
        data_tuple_list.append(data_tuple)
    
    return tuple(data_tuple_list[0])