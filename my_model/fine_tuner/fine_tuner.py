
#   Main Fine-Tuning Script for meta-llama/Llama-2-7b-chat-hf

#   This script is the central executable for fine-tuning large language models, specifically designed for the LLAMA2
#   model.
#   It encompasses the entire process of fine-tuning, starting from data preparation to the final model training.
#   The script leverages the 'FinetuningDataHandler' class for data loading, inspection, preparation, and splitting.
#   This ensures that the dataset is correctly processed and prepared for effective training.

#   The fine-tuning process is managed by the Finetuner class, which handles the training of the model using specific
#   training arguments and datasets. Advanced configurations for Quantized Low-Rank Adaptation (QLoRA) and Parameter
#   Efficient Fine-Tuning (PEFT) are utilized to optimize the training process on limited hardware resources.

#   The script is designed to be executed as a standalone process, providing an end-to-end solution for fine-tuning
#   LLMs. It is a part of a larger project aimed at optimizing the performance of language model to adapt to
#   OK-VQA dataset.

#   Ensure all dependencies are installed and the required files are in place before running this script.
#   The configurations for the fine-tuning process are defined in the 'fine_tuning_config.py' file.

#   ---------- Please run this file for the full fine-tuning process to start ----------#
#   ---------- Please ensure this is run on a GPU ----------#


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TRANSFORMERS_CACHE
from trl import SFTTrainer
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
import fine_tuning_config as config
from typing import List
import bitsandbytes  # only on GPU
import gc
import os
import shutil
from my_model.LLAMA2.LLAMA2_model import Llama2ModelManager
from fine_tuning_data_handler import FinetuningDataHandler


class QLoraConfig:
    """
    Configures QLoRA (Quantized Low-Rank Adaptation) parameters for efficient model fine-tuning.
    LoRA allows adapting large language models with a minimal number of trainable parameters.

    Attributes:
        lora_config (LoraConfig): Configuration object for LoRA parameters.
    """

    def __init__(self) -> None:
        """
        Initializes QLoraConfig with specific LoRA parameters.

        """
        # please refer to config file 'fine_tuning_config.py' for QLORA arguments description.
        self.lora_config = LoraConfig(
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            r=config.LORA_R,
            bias="none",  # bias is already accounted for in LLAMA2 pre-trained model layers.
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']  # modules for fine-tuning.
        )


class Finetuner:
    """
       The Finetuner class manages the fine-tuning process of a pre-trained language model using specific
       training arguments and datasets. It is designed to adapt a pre-trained model on a specific dataset
       to enhance its performance on similar data.

       This class not only facilitates the fine-tuning of LLAMA2 but also includes advanced
       resource management capabilities. It provides methods for deleting model and trainer objects,
       clearing GPU memory, and cleaning up Hugging Face's Transformers cache. These functionalities
       make the Finetuner class especially useful in environments with limited computational resources
       or when managing multiple models or training sessions.

       Additionally, the class supports configurations for Quantized Low-Rank Adaptation (QLoRA)
       to fine-tune models with minimal trainable parameters, and Parameter Efficient Fine-Tuning (PEFT)
       for training efficiency on limited hardware.

       Attributes:
           base_model (AutoModelForCausalLM): The pre-trained language model to be fine-tuned.
           tokenizer (AutoTokenizer): The tokenizer associated with the model.
           train_dataset (Dataset): The dataset used for training.
           eval_dataset (Dataset): The dataset used for evaluation.
           training_arguments (TrainingArguments): Configuration for training the model.

       Key Methods:
           - load_LLAMA2_for_finetuning: Loads the LLAMA2 model and tokenizer for fine-tuning.
           - train: Trains the model using PEFT configuration.
           - delete_model: Deletes a specified model attribute.
           - delete_trainer: Deletes a specified trainer object.
           - clear_training_resources: Clears GPU memory.
           - clear_cache_and_collect_garbage: Clears Transformers cache and performs garbage collection.
           - find_all_linear_names: Identifies linear layer names suitable for LoRA application.
           - print_trainable_parameters: Prints the number of trainable parameters in the model.
       """

    def __init__(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        """
        Initializes the Finetuner class with the model, tokenizer, and datasets.

        Args:
            model (AutoModelForCausalLM): The pre-trained language model.
            tokenizer (AutoTokenizer): The tokenizer for the model.
            train_dataset (Dataset): The dataset for training the model.
            eval_dataset (Dataset): The dataset for evaluating the model.
        """

        self.base_model, self.tokenizer = self.load_LLAMA2_for_finetuning()
        self.merged_model = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # please refer to config file 'fine_tuning_config.py' for training arguments description.
        self.training_arguments = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=config.NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            fp16=config.FP16,
            bf16=config.BF16,
            evaluation_strategy=config.Evaluation_STRATEGY,
            eval_steps=config.EVALUATION_STEPS,
            max_grad_norm=config.MAX_GRAD_NORM,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            optim=config.OPTIM,
            lr_scheduler_type=config.LR_SCHEDULER_TYPE,
            max_steps=config.MAX_STEPS,
            warmup_ratio=config.WARMUP_RATIO,
            group_by_length=config.GROUP_BY_LENGTH,
            save_steps=config.SAVE_STEPS,
            logging_steps=config.LOGGING_STEPS,
            report_to="tensorboard"
        )

    def load_LLAMA2_for_finetuning(self):
        """
        Loads the LLAMA2 model and tokenizer, specifically configured for fine-tuning.
        This method ensures the model is ready to be adapted to a specific task or dataset.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
        """

        llm_manager = Llama2ModelManager()
        base_model, tokenizer = llm_manager.load_model_and_tokenizer(for_fine_tuning=True)

        return base_model, tokenizer

    def find_all_linear_names(self) -> List[str]:
        """
        Identifies all linear layer names in the model that are suitable for applying LoRA.

        Returns:
            List[str]: A list of linear layer names.
        """
        cls = bitsandbytes.nn.Linear4bit
        lora_module_names = set()
        for name, module in self.base_model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # We dont want to train these two modules to avoid computational overhead.
        lora_module_names -= {'lm_head', 'gate_proj'}
        return list(lora_module_names)

    def print_trainable_parameters(self, use_4bit: bool = False) -> None:
        """
        Calculates and prints the number of trainable parameters in the model.

        Args:
            use_4bit (bool): If true, calculates the parameter count considering 4-bit quantization.
        """
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        if use_4bit:
            trainable_params /= 2

        total_params = sum(p.numel() for p in self.base_model.parameters())
        print(f"All Parameters: {total_params:,d} || Trainable Parameters: {trainable_params:,d} "
              f"|| Trainable Parameters %: {100 * trainable_params / total_params:.2f}%")

    def train(self, peft_config: LoraConfig) -> None:
        """
        Trains the model using the specified PEFT (Progressive Effort Fine-Tuning) configuration.

        Args:
            peft_config (LoraConfig): Configuration for the PEFT training process.
        """
        self.base_model.config.use_cache = False
        # Set the pretraining_tp flag to 1 to enable the use of LoRA (Low-Rank Adapters) layers.
        self.base_model.config.pretraining_tp = 1
        # Prepare the model for k-bit training by quantizing the weights to 4 bits using bitsandbytes.
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=peft_config,
            dataset_text_field='text',
            max_seq_length=config.MAX_TOKEN_COUNT,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=config.PACKING
        )
        self.trainer.train()

    def save_model(self):

        """
        Saves the fine-tuned model to the specified directory.

        This method saves the model weights and configuration of the fine-tuned model.
        The save directory and filename are determined by the configuration provided in
        the 'fine_tuning_config.py' file. It is useful for persisting the fine-tuned model
        for later use or evaluation.

        The saved model can be easily loaded using Hugging Face's model loading utilities.
        """

        self.fine_tuned_adapter_name = config.ADAPTER_SAVE_NAME
        self.trainer.model.save_pretrained(self.fine_tuned_adapter_name)

    def merge_weights(self):
        """
        Merges the weights of the fine-tuned adapter with the base model.

        This method integrates the fine-tuned adapter weights into the base model,
        resulting in a single consolidated model. The merged model can then be used
        for inference or further training.

        After merging, the weights of the adapter are no longer separate from the
        base model, enabling more efficient storage and deployment. The merged model
        is stored in the 'self.merged_model' attribute of the Finetuner class.
        """

        self.merged_model = PeftModel.from_pretrained(self.base_model, self.fine_tuned_adapter_name)
        self.merged_model = self.merged_model.merge_and_unload()

    def delete_model(self, model_name: str):
        """
        Deletes a specified model attribute.

        Args:
            model_name (str): The name of the model attribute to delete.
        """
        try:
            if hasattr(self, model_name) and getattr(self, model_name) is not None:
                delattr(self, model_name)
                print(f"Model '{model_name}' has been deleted.")
            else:
                print(f"Warning: Model '{model_name}' has already been cleared or does not exist.")
        except Exception as e:
            print(f"Error occurred while deleting model '{model_name}': {str(e)}")

    def delete_trainer(self, trainer_name: str):
        """
        Deletes a specified trainer object.

        Args:
            trainer_name (str): The name of the trainer object to delete.
        """
        try:
            if hasattr(self, trainer_name) and getattr(self, trainer_name) is not None:
                delattr(self, trainer_name)
                print(f"Trainer object '{trainer_name}' has been deleted.")
            else:
                print(f"Warning: Trainer object '{trainer_name}' has already been cleared or does not exist.")
        except Exception as e:
            print(f"Error occurred while deleting trainer object '{trainer_name}': {str(e)}")

    def clear_training_resources(self):
        """
        Clears GPU memory.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU memory has been cleared.")
        except Exception as e:
            print(f"Error occurred while clearing GPU memory: {str(e)}")

    def clear_cache_and_collect_garbage(self):
        """
        Clears Hugging Face's Transformers cache and runs garbage collection.
        """
        try:
            if os.path.exists(TRANSFORMERS_CACHE):
                shutil.rmtree(TRANSFORMERS_CACHE, ignore_errors=True)
                print("Transformers cache has been cleared.")

            gc.collect()
            print("Garbage collection has been executed.")
        except Exception as e:
            print(f"Error occurred while clearing cache and collecting garbage: {str(e)}")

def fine_tune(save_fine_tuned_adapter=False, merge=False, delete_trainer_after_fine_tune=False):
    """
    Conducts the fine-tuning process of a pre-trained language model using specified configurations.
    This function encompasses the complete workflow of fine-tuning, including data handling, training,
    and optional steps like saving the fine-tuned model and merging weights.

    Args:
        save_fine_tuned_adapter (bool): If True, saves the fine-tuned adapter after training.
        merge (bool): If True, merges the weights of the fine-tuned adapter into the base model.
        delete_trainer_after_fine_tune (bool): If True, deletes the trainer object after fine-tuning to free up resources.

    Returns:
        The fine-tuned model after the fine-tuning process. This could be either the merged model
        or the trained model based on the provided arguments.

    The function initiates by preparing the training and evaluation datasets using the `FinetuningDataHandler`.
    It then sets up the QLoRA configuration for the fine-tuning process. The actual training is carried out by
    the `Finetuner` class. Post training, based on the arguments, the function can save the fine-tuned model,
    merge the adapter weights with the base model, and clean up resources by deleting the trainer object.
    """

    data_handler = FinetuningDataHandler()
    fine_tuning_data_train, fine_tuning_data_eval = data_handler.inspect_prepare_split_data()
    qlora = QLoraConfig()
    peft_config = qlora.lora_config
    tuner = Finetuner(fine_tuning_data_train, fine_tuning_data_eval)
    tuner.train(peft_config=peft_config)
    if save_fine_tuned_adapter:
        tuner.save_model()

    if merge:
        tuner.merge_weights()

    if delete_trainer_after_fine_tune:
        tuner.delete_trainer("trainer")

    tuner.delete_model("base_model")  # We always delete this as it is not required after the merger.

    if save_fine_tuned_adapter:
        tuner.save_model()
        if tuner.merged_model is not None:
            return tuner.merged_model
        else:
            return tuner.trainer.model



if __name__ == "__main__":
    fine_tune()
