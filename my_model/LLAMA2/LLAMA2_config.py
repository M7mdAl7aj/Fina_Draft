# Configuration parameters for LLaMA-2 model
import torch

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
TOKENIZER_NAME = "meta-llama/Llama-2-7b-chat-hf"
QUANTIZATION = '4bit'  # Options: '4bit', '8bit', or None
FROM_SAVED = False
MODEL_PATH = None
TRUST_REMOTE = False
USE_FAST = True
ADD_EOS_TOKEN = True
ACCESS_TOKEN = "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"  # My HF Read only Token
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
