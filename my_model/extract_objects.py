from object_detection import ObjectDetector
import os

def detect_objects_for_image(image_name, detector):

    if os.path.exists(image_path):
        image = detector.process_image(image_path)
        detected_objects_str, _ = detector.detect_objects(image)
        return detected_objects_str
    else:
        return "Image not found"

def add_detected_objects_to_dataframe(df, image_directory, detector):
    """
    Adds a column to the DataFrame with detected objects for each image specified in the 'image_name' column.

    Parameters:
    df (pd.DataFrame): DataFrame containing a column 'image_name' with image filenames.
    image_directory (str): Path to the directory containing images.
    detector (ObjectDetector): An instance of the ObjectDetector class.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column 'detected_objects'.
    """

    # Ensure 'image_name' column exists in the DataFrame
    if 'image_name' not in df.columns:
        raise ValueError("DataFrame must contain an 'image_name' column.")

    image_path = os.path.join(image_directory, image_name)

    # Function to detect objects for a given image filename


    # Apply the function to each row in the DataFrame
    df['detected_objects'] = df['image_name'].apply(detect_objects_for_image)

    return df

# Example usage (assuming the function will be used in a context where 'detector' is defined and configured):
# df_images = pd.DataFrame({"image_name": ["image1.jpg", "image2.jpg", ...]})
# image_directory = "path/to/image_directory"
# updated_df = add_detected_objects_to_dataframe(df_images, image_directory, detector)
# updated_df.head()

