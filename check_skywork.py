import torch
import random
from typing import Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Global variables for model and tokenizer (loaded once)
_skywork_model = None
_skywork_tokenizer = None
# _device = "cuda:5"
_model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

def load_skywork_model():
    """Load Skywork model and tokenizer if not already loaded"""
    global _skywork_model, _skywork_tokenizer
    
    if _skywork_model is None:
        print(f"Loading {_model_name}")
        _skywork_model = AutoModelForSequenceClassification.from_pretrained(
            _model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        _skywork_tokenizer = AutoTokenizer.from_pretrained(_model_name)
        device = next(iter(_skywork_model.parameters())).device
        print(f"{_model_name} loaded successfully")
        print(f"Loading {_model_name} on device {device}")

def compute_skywork_reward_in_batch(conversations: List[Dict]) -> List[Dict]:
    """
    Compute Skywork reward for a batch of conversations.
    
    Args:
        conversations: List of conversations
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    """
    try:
        load_skywork_model()

        conv_formatted_batch = [_skywork_tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]

        if _skywork_tokenizer.bos_token is not None:
            conv_formatted_batch = [
                s[len(_skywork_tokenizer.bos_token):] if s.startswith(_skywork_tokenizer.bos_token) else s
                for s in conv_formatted_batch
            ]

        batch = _skywork_tokenizer(conv_formatted_batch, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            logits = _skywork_model(**batch).logits.squeeze(-1)
            scores = logits.tolist()

        return scores
    
    except Exception as e:
        print(f"Error in Skywork reward computation in batch: {e}")
        return None
    

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
convs = [conv for _ in range(100)]
scores = compute_skywork_reward_in_batch(convs)
print(scores)