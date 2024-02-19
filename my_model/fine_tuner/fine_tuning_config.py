# Configurable parameters for fine-tuning

import os


# *** Dataset ***
# Base directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the folder containing the data files, relative to the configuration file
DATA_FOLDER = 'fine_tuning_data'
# Full path to the data folder
DATA_FOLDER_PATH = os.path.join(BASE_DIR, DATA_FOLDER)
# Path to the dataset file (CSV format)
DATASET_FILE = os.path.join(DATA_FOLDER_PATH, 'fine_tuning_data_yolov5.csv')  # or 'fine_tuning_data_detic.csv'


# *** Fine-tuned Adapter ***
TRAINED_ADAPTER_NAME = 'fine_tuned_adapter'  # name of fine-tuned adapter.
FINE_TUNED_ADAPTER_FOLDER = 'fine_tuned_model'
FINE_TUNED_ADAPTER_PATH = os.path.join(BASE_DIR, FINE_TUNED_ADAPTER_FOLDER)
ADAPTER_SAVE_NAME = os.path.join(FINE_TUNED_ADAPTER_PATH, TRAINED_ADAPTER_NAME)


# Proportion of the dataset to include in the test split (e.g., 0.1 for 10%)
TEST_SIZE = 0.1

# Seed for random operations to ensure reproducibility
SEED = 123

# *** QLoRA Configuration Parameters ***
# LoRA attention dimension: number of additional parameters in each LoRA layer
LORA_R = 64

# Alpha parameter for LoRA scaling: controls the scaling of LoRA weights
LORA_ALPHA = 32

# Dropout probability for LoRA layers: probability of dropping a unit in LoRA layers
LORA_DROPOUT = 0.05



# *** TrainingArguments Configuration Parameters for the Transformers library ***
# Output directory to save model predictions and checkpoints
OUTPUT_DIR = "./TUNED_MODEL_LLAMA"

# Number of epochs to train the model
NUM_TRAIN_EPOCHS = 1

# Enable mixed-precision training using fp16 (set to True for faster training)
FP16 = True

# Enable mixed-precision training using bf16 (set to True if using an A100 GPU)
BF16 = False

# Batch size per GPU/Device for training
PER_DEVICE_TRAIN_BATCH_SIZE = 16

# Batch size per GPU/Device for evaluation
PER_DEVICE_EVAL_BATCH_SIZE = 8

# Number of update steps to accumulate gradients before performing a backward/update pass
GRADIENT_ACCUMULATION_STEPS = 1

# Enable gradient checkpointing to reduce memory usage at the cost of a slight slowdown
GRADIENT_CHECKPOINTING = True

# Maximum gradient norm for gradient clipping to prevent exploding gradients
MAX_GRAD_NORM = 0.3

# Initial learning rate for the AdamW optimizer
LEARNING_RATE = 2e-4

# Weight decay coefficient for regularization (applied to all layers except bias/LayerNorm weights)
WEIGHT_DECAY = 0.01

# Optimizer type, here using 'paged_adamw_8bit' for efficient training
OPTIM = "paged_adamw_8bit"

# Learning rate scheduler type (e.g., 'linear', 'cosine', etc.)
LR_SCHEDULER_TYPE = "linear"

# Maximum number of training steps, overrides 'num_train_epochs' if set to a positive number
# Setting MAX_STEPS = -1 in training arguments for SFTTrainer means that the number of steps will be determined by the
# number of epochs, the size of the dataset, the batch size, and the number of GPUs1. This is the default behavior
# when MAX_STEPS is not specified or set to a negative value2.
MAX_STEPS = -1

# Ratio of the total number of training steps used for linear warmup
WARMUP_RATIO = 0.03

# Whether to group sequences into batches with the same length to save memory and increase speed
GROUP_BY_LENGTH = False

# Save a model checkpoint every X update steps
SAVE_STEPS = 50

# Log training information every X update steps
LOGGING_STEPS = 25

PACKING = False

# Evaluation strategy during training ("steps", "epoch, "no")
Evaluation_STRATEGY = "steps"

# Number of update steps between two evaluations if `evaluation_strategy="steps"`.
# Will default to the same value as `logging_steps` if not set.
EVALUATION_STEPS = 5

# Maximum number of tokens per sample in the dataset
MAX_TOKEN_COUNT = 1024


if __name__=="__main__":
    pass