from my_model.utilities import is_pycharm
import seaborn as sns
from transformers import AutoTokenizer
from datasets import Dataset, load_dataset
import fine_tuning_config as config
from my_model.LLAMA2.LLAMA2_model import Llama2ModelManager
from typing import Tuple



class FinetuningDataHandler:
    """
    A class dedicated to handling data for fine-tuning language models. It manages loading,
    inspecting, preparing, and splitting the dataset, specifically designed to filter out
    data samples exceeding a specified token count limit. This is crucial for models with
    token count constraints and it helps control the level of GPU RAM tolernace based on the number of tokens,
    ensuring efficient and effective model fine-tuning.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer used for tokenizing the dataset.
        dataset_file (str): File path to the dataset.
        max_token_count (int): Maximum allowable token count per data sample.

    Methods:
        load_llm_tokenizer(): Loads the LLM tokenizer and adds special tokens, if not already loaded.
        load_dataset(): Loads the dataset from a specified file path.
        plot_tokens_count_distribution(token_counts, title): Plots the distribution of token counts in the dataset.
        filter_dataset_by_indices(dataset, valid_indices): Filters the dataset based on valid indices, removing samples exceeding token limits.
        get_token_counts(dataset): Calculates token counts for each sample in the dataset.
        prepare_dataset(): Tokenizes and filters the dataset, preparing it for training. Also visualizes token count distribution before and after filtering.
        split_dataset_for_train_eval(dataset): Divides the dataset into training and evaluation sets.
        inspect_prepare_split_data(): Coordinates the data preparation and splitting process for fine-tuning.
    """

    def __init__(self, tokenizer: AutoTokenizer = None, dataset_file: str = config.DATASET_FILE) -> None:
        """
        Initializes the FinetuningDataHandler class.

        Args:
            tokenizer (AutoTokenizer): Tokenizer to use for tokenizing the dataset.
            dataset_file (str): Path to the dataset file.
        """
        self.tokenizer = tokenizer  # The tokenizer used for processing the dataset.
        self.dataset_file = dataset_file  # Path to the fine-tuning dataset file.
        self.max_token_count = config.MAX_TOKEN_COUNT  # Max token count for filtering.

    def load_llm_tokenizer(self):
        """
        Loads the LLM tokenizer and adds special tokens, if not already loaded.
        If the tokenizer is already loaded, this method does nothing.
        """

        if self.tokenizer is None:
            llm_manager = Llama2ModelManager()  # Initialize Llama2 model manager.
            # we only need the tokenizer for the data inspection not the model itself.
            self.tokenizer = llm_manager.load_tokenizer()
            llm_manager.add_special_tokens()  # Add special tokens specific to LLAMA2 vocab for efficient tokenization.

    def load_dataset(self) -> Dataset:
        """
        Loads the dataset from the specified file path. The dataset is expected to be in CSV format.

        Returns:
            Dataset: The loaded dataset, ready for processing.
        """
        return load_dataset('csv', data_files=self.dataset_file)

    def plot_tokens_count_distribution(self, token_counts: list, title: str = "Token Count Distribution") -> None:
        """
        Plots the distribution of token counts in the dataset for visualization purposes.

        Args:
            token_counts (list): List of token counts, each count representing the number of tokens in a dataset sample.
            title (str): Title for the plot, highlighting the nature of the distribution.
        """

        if is_pycharm():  # Ensuring compatibility with PyCharm's environment for interactive plot.
            import matplotlib
            matplotlib.use('TkAgg')  # Set the backend to 'TkAgg'
        import matplotlib.pyplot as plt
        sns.set_style("whitegrid")
        plt.figure(figsize=(15, 6))
        plt.hist(token_counts, bins=50, color='#3498db', edgecolor='black')
        plt.title(title, fontsize=16)
        plt.xlabel("Number of Tokens", fontsize=14)
        plt.ylabel("Number of Samples", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def filter_dataset_by_indices(self, dataset: Dataset, valid_indices: list) -> Dataset:
        """
        Filters the dataset based on a list of valid indices. This method is used to exclude
        data samples that have a token count exceeding the specified maximum token count.

        Args:
            dataset (Dataset): The dataset to be filtered.
            valid_indices (list): Indices of samples with token counts within the limit.

        Returns:
            Dataset: Filtered dataset containing only samples with valid indices.
        """
        return dataset['train'].select(valid_indices)  # Select only samples with valid indices based on token count.

    def get_token_counts(self, dataset):
        """
        Calculates and returns the token counts for each sample in the dataset.
        This function assumes the dataset has a 'train' split and a 'text' field.

        Args:
            dataset (Dataset): The dataset for which to count tokens.

        Returns:
            List[int]: List of token counts per sample in the dataset.
        """

        if 'train' in dataset:
            return [len(self.tokenizer.tokenize(s)) for s in dataset["train"]["text"]]
        else:
            # After filtering the samples with unacceptable token count, the dataset is already
            # dataset = dataset['train']
            return [len(self.tokenizer.tokenize(s)) for s in dataset["text"]]

    def prepare_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Prepares the dataset for fine-tuning by tokenizing the data and filtering out samples
        that exceed the maximum used context window (configurable through max_token_count).
        It also visualizes the token count distribution before and after filtering.

        Returns:
            Tuple[Dataset, Dataset]: The train and evaluate datasets, post-filtering.
        """
        dataset = self.load_dataset()
        self.load_llm_tokenizer()

        # Count tokens in each dataset sample before filtering
        token_counts_before_filtering = self.get_token_counts(dataset)
        # Plot token count distribution before filtering for visualization.
        self.plot_tokens_count_distribution(token_counts_before_filtering, "Token Count Distribution Before Filtration")
        # Identify valid indices based on max token count.
        valid_indices = [i for i, count in enumerate(token_counts_before_filtering) if count <= self.max_token_count]
        # Filter the dataset to exclude samples with excessive token counts.
        filtered_dataset = self.filter_dataset_by_indices(dataset, valid_indices)

        token_counts_after_filtering = self.get_token_counts(filtered_dataset)
        self.plot_tokens_count_distribution(token_counts_after_filtering, "Token Count Distribution After Filtration")

        return self.split_dataset_for_train_eval(filtered_dataset)  # split the dataset into training and evaluation.

    def split_dataset_for_train_eval(self, dataset) -> Tuple[Dataset, Dataset]:
        """
        Splits the dataset into training and evaluation datasets.

        Args:
            dataset (Dataset): The dataset to split.

        Returns:
            tuple[Dataset, Dataset]: The split training and evaluation datasets.
        """
        split_data = dataset.train_test_split(test_size=config.TEST_SIZE, shuffle=True, seed=config.SEED)
        train_data, eval_data = split_data['train'], split_data['test']
        return train_data, eval_data

    def inspect_prepare_split_data(self) -> tuple[Dataset, Dataset]:
        """
        Orchestrates the process of inspecting, preparing, and splitting the dataset for fine-tuning.

        Returns:
            tuple[Dataset, Dataset]: The prepared training and evaluation datasets.
        """
        return self.prepare_dataset()


# Example usage
if __name__ == "__main__":

    #  Please uncomment the below lines to test the data prep.
    #data_handler = FinetuningDataHandler()
    #fine_tuning_data_train, fine_tuning_data_eval = data_handler.inspect_prepare_split_data()
    #print(fine_tuning_data_train, fine_tuning_data_eval)
    pass
