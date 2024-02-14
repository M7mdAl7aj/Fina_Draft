import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import bitsandbytes  # only for using on GPU
import accelerate  # only for using on GPU
import LLAMA2_config as config  # Importing LLAMA2 configuration file
import warnings

# Suppress only FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


class Llama2ModelManager:
    """
    Manages loading and configuring the LLaMA-2 model and tokenizer.

    Attributes:
        device (str): Device to use for the model ('cuda' or 'cpu').
        model_name (str): Name or path of the pre-trained model.
        tokenizer_name (str): Name or path of the tokenizer.
        quantization (str): Specifies the quantization level ('4bit', '8bit', or None).
        from_saved (bool): Flag to load the model from a saved path.
        model_path (str or None): Path to the saved model if `from_saved` is True.
        trust_remote (bool): Whether to trust remote code when loading the tokenizer.
        use_fast (bool): Whether to use the fast version of the tokenizer.
        add_eos_token (bool): Whether to add an EOS token to the tokenizer.
        access_token (str): Access token for Hugging Face Hub.
        model (AutoModelForCausalLM or None): Loaded model, initially None.
    """

    def __init__(self) -> None:
        """
        Initializes the Llama2ModelManager class with configuration settings.
        """
        self.device: str = config.DEVICE
        self.model_name: str = config.MODEL_NAME
        self.tokenizer_name: str = config.TOKENIZER_NAME
        self.quantization: str = config.QUANTIZATION
        self.from_saved: bool = config.FROM_SAVED
        self.model_path: Optional[str] = config.MODEL_PATH
        self.trust_remote: bool = config.TRUST_REMOTE
        self.use_fast: bool = config.USE_FAST
        self.add_eos_token: bool = config.ADD_EOS_TOKEN
        self.access_token: str = config.ACCESS_TOKEN
        self.model: Optional[AutoModelForCausalLM] = None

    def create_bnb_config(self) -> BitsAndBytesConfig:
        """
        Creates a BitsAndBytes configuration based on the quantization setting.

        Returns:
            BitsAndBytesConfig: Configuration for BitsAndBytes optimized model.
        """
        if self.quantization == '4bit':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif self.quantization == '8bit':
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_compute_dtype=torch.bfloat16
            )

    def load_model(self) -> AutoModelForCausalLM:
        """
        Loads the LLaMA-2 model based on the specified configuration. If the model is already loaded, returns the existing model.

        Returns:
            AutoModelForCausalLM: Loaded LLaMA-2 model.
        """
        if self.model is not None:
            print("Model is already loaded.")
            return self.model

        if self.from_saved:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        else:
            bnb_config = None if self.quantization is None else self.create_bnb_config()
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto",
                                                              quantization_config=bnb_config,
                                                              torch_dtype=torch.float16,
                                                              token=self.access_token)

        if self.model is not None:
            print(f"LLAMA2 Model loaded successfully in {self.quantization} quantization.")
        else:
            print("LLAMA2 Model failed to load.")
        return self.model

    def load_tokenizer(self) -> AutoTokenizer:
        """
        Loads the tokenizer for the LLaMA-2 model with the specified configuration.

        Returns:
            AutoTokenizer: Loaded tokenizer for LLaMA-2 model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=self.use_fast,
                                                       token=self.access_token,
                                                       trust_remote_code=self.trust_remote,
                                                       add_eos_token=self.add_eos_token)

        if self.tokenizer is not None:
            print(f"LLAMA2 Tokenizer loaded successfully.")
        else:
            print("LLAMA2 Tokenizer failed to load.")

        return self.tokenizer

    def add_special_tokens(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                           tokens: Optional[list[str]] = None) -> None:
        """
        Adds special tokens to the tokenizer and updates the model's token embeddings.

        Args:
            model (AutoModelForCausalLM): The LLaMA-2 model.
            tokenizer (AutoTokenizer): The tokenizer for the model.
            tokens (list of str, optional): Special tokens to add. Defaults to a predefined set.

        Returns:
            None
        """
        if tokens is None:
            tokens = ['[CAP]', '[/CAP]', '[QES]', '[/QES]', '[OBJ]', '[/OBJ]']
        print(f'Model and Tokenizer vocabulary size is: {len(tokenizer)}')
        print(f"Adding the following tokens: {tokens}")
        tokenizer.add_tokens(tokens, special_tokens=True)

        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        print(f"Adding Padding Token {tokenizer.pad_token}")
        tokenizer.padding_side = "right"
        print(f'Padding side: {tokenizer.padding_side}')

        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

        print(f'New Vocabulary Size for Model and Tokenizer is: {len(tokenizer)}')
        print(f'Padding Token is: {tokenizer.pad_token}')
        print(f'Special Tokens: {tokenizer.added_tokens_decoder}')


if __name__ == "__main__":
    LLAMA2_manager = Llama2ModelManager()
    LLAMA2_model = LLAMA2_manager.load_model()  # First time loading the model
    LLAMA2_tokenizer = LLAMA2_manager.load_tokenizer()
    LLAMA2_manager.add_special_tokens(LLAMA2_model, LLAMA2_tokenizer)
