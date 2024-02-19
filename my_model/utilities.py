import pandas as pd
from collections import Counter
import json
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import get_ipython
import sys


class VQADataProcessor:
    """
    A class to process OKVQA dataset.

    Attributes:
        questions_file_path (str): The file path for the questions JSON file.
        annotations_file_path (str): The file path for the annotations JSON file.
        questions (list): List of questions extracted from the JSON file.
        annotations (list): List of annotations extracted from the JSON file.
        df_questions (DataFrame): DataFrame created from the questions list.
        df_answers (DataFrame): DataFrame created from the annotations list.
        merged_df (DataFrame): DataFrame resulting from merging questions and answers.
    """

    def __init__(self, questions_file_path, annotations_file_path):
        """
        Initializes the VQADataProcessor with file paths for questions and annotations.

        Parameters:
            questions_file_path (str): The file path for the questions JSON file.
            annotations_file_path (str): The file path for the annotations JSON file.
        """
        self.questions_file_path = questions_file_path
        self.annotations_file_path = annotations_file_path
        self.questions, self.annotations = self.read_json_files()
        self.df_questions = pd.DataFrame(self.questions)
        self.df_answers = pd.DataFrame(self.annotations)
        self.merged_df = None

    def read_json_files(self):
        """
        Reads the JSON files for questions and annotations.

        Returns:
            tuple: A tuple containing two lists: questions and annotations.
        """
        with open(self.questions_file_path, 'r') as file:
            data = json.load(file)
            questions = data['questions']

        with open(self.annotations_file_path, 'r') as file:
            data = json.load(file)
            annotations = data['annotations']

        return questions, annotations

    @staticmethod
    def find_most_frequent(my_list):
        """
        Finds the most frequent item in a list.

        Parameters:
            my_list (list): A list of items.

        Returns:
            The most frequent item in the list. Returns None if the list is empty.
        """
        if not my_list:
            return None
        counter = Counter(my_list)
        most_common = counter.most_common(1)
        return most_common[0][0]

    def merge_dataframes(self):
        """
        Merges the questions and answers DataFrames on 'question_id' and 'image_id'.
        """
        self.merged_df = pd.merge(self.df_questions, self.df_answers, on=['question_id', 'image_id'])

    def join_words_with_hyphen(self, sentence):

        return '-'.join(sentence.split())

    def process_answers(self):
        """
        Processes the answers by extracting raw and processed answers and finding the most frequent ones.
        """
        if self.merged_df is not None:
            self.merged_df['raw_answers'] = self.merged_df['answers'].apply(lambda x: [ans['raw_answer'] for ans in x])
            self.merged_df['processed_answers'] = self.merged_df['answers'].apply(
                lambda x: [ans['answer'] for ans in x])
            self.merged_df['most_frequent_raw_answer'] = self.merged_df['raw_answers'].apply(self.find_most_frequent)
            self.merged_df['most_frequent_processed_answer'] = self.merged_df['processed_answers'].apply(
                self.find_most_frequent)
            self.merged_df.drop(columns=['answers'], inplace=True)
        else:
            print("DataFrames have not been merged yet.")

        # Apply the function to the 'most_frequent_processed_answer' column
        self.merged_df['single_word_answers'] = self.merged_df['most_frequent_processed_answer'].apply(
            self.join_words_with_hyphen)

    def get_processed_data(self):
        """
        Retrieves the processed DataFrame.

        Returns:
            DataFrame: The processed DataFrame. Returns None if the DataFrame is empty or not processed.
        """
        if self.merged_df is not None:
            return self.merged_df
        else:
            print("DataFrame is empty or not processed yet.")
            return None

    def save_to_csv(self, df, saved_file_name):

        if saved_file_name is not None:
            if ".csv" not in saved_file_name:
                df.to_csv(os.path.join(saved_file_name, ".csv"), index=None)

            else:
                df.to_csv(saved_file_name, index=None)

        else:
            df.to_csv("data.csv", index=None)

    def display_dataframe(self):
        """
        Displays the processed DataFrame.
        """
        if self.merged_df is not None:
            print(self.merged_df)
        else:
            print("DataFrame is empty.")


def process_okvqa_dataset(questions_file_path, annotations_file_path, save_to_csv=False, saved_file_name=None):
    """
    Processes the OK-VQA dataset given the file paths for questions and annotations.

    Parameters:
        questions_file_path (str): The file path for the questions JSON file.
        annotations_file_path (str): The file path for the annotations JSON file.

    Returns:
        DataFrame: The processed DataFrame containing merged and processed VQA data.
    """
    # Create an instance of the class
    processor = VQADataProcessor(questions_file_path, annotations_file_path)

    # Process the data
    processor.merge_dataframes()
    processor.process_answers()

    # Retrieve the processed DataFrame
    processed_data = processor.get_processed_data()

    if save_to_csv:
        processor.save_to_csv(processed_data, saved_file_name)

    return processed_data


def show_image(image):
    """
    Display an image in various environments (Jupyter, PyCharm, Hugging Face Spaces).
    Handles different types of image inputs (file path, PIL Image, numpy array, OpenCV, PyTorch tensor).

    Args:
    image (str or PIL.Image or numpy.ndarray or torch.Tensor): The image to display.
    """
    in_jupyter = is_jupyter_notebook()
    in_colab = is_google_colab()

    # Convert image to PIL Image if it's a file path, numpy array, or PyTorch tensor
    if isinstance(image, str):

        if os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError("File path provided does not exist.")
    elif isinstance(image, np.ndarray):

        if image.ndim == 3 and image.shape[2] in [3, 4]:

            image = Image.fromarray(image[..., ::-1] if image.shape[2] == 3 else image)
        else:

            image = Image.fromarray(image)
    elif torch.is_tensor(image):

        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))

    # Display the image
    if in_jupyter or in_colab:

        from IPython.display import display
        display(image)
    else:

        image.show()



def show_image_with_matplotlib(image):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif torch.is_tensor(image):
        image = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))

    plt.imshow(image)
    plt.axis('off')  # Turn off axis numbers
    plt.show()


def is_jupyter_notebook():
    """
    Check if the code is running in a Jupyter notebook.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
        if 'ipykernel' in str(type(get_ipython())):
            return True  # Running in Jupyter Notebook
    except (NameError, AttributeError):
        return False  # Not running in Jupyter Notebook

    return False  # Default to False if none of the above conditions are met


def is_pycharm():
    return 'PYCHARM_HOSTED' in os.environ


def is_google_colab():
    return 'COLAB_GPU' in os.environ or 'google.colab' in sys.modules


def get_path(name, path_type):
    """
    Generates a path for models, images, or data based on the specified type.

    Args:
    name (str): The name of the model, image, or data folder/file.
    path_type (str): The type of path needed ('models', 'images', or 'data').

    Returns:
    str: The full path to the specified resource.
    """
    # Get the current working directory (assumed to be inside 'code' folder)
    current_dir = os.getcwd()

    # Get the directory one level up (the parent directory)
    parent_dir = os.path.dirname(current_dir)

    # Construct the path to the specified folder
    folder_path = os.path.join(parent_dir, path_type)

    # Construct the full path to the specific resource
    full_path = os.path.join(folder_path, name)

    return full_path



if __name__ == "__main__":
    pass
    #val_data = process_okvqa_dataset('OpenEnded_mscoco_val2014_questions.json', 'mscoco_val2014_annotations.json', save_to_csv=True, saved_file_name="okvqa_val.csv")
    #train_data = process_okvqa_dataset('OpenEnded_mscoco_train2014_questions.json', 'mscoco_train2014_annotations.json', save_to_csv=True, saved_file_name="okvqa_train.csv")
