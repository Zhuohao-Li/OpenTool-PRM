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

def compute_skywork_reward(question: str, answer: str) -> Dict:
    """
    Compute Skywork reward score for a question-answer pair.
    
    Args:
        question: The question/prompt string
        answer: The answer/response string
        
    Returns:
        Dict containing score and max_score, or None if error
    """
    try:
        # Ensure model is loaded
        load_skywork_model()
        
        # Format conversation as expected by Skywork model
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        conv_formatted = _skywork_tokenizer.apply_chat_template(conversation, tokenize=False)
        if _skywork_tokenizer.bos_token is not None and conv_formatted.startswith(_skywork_tokenizer.bos_token):
            conv_formatted = conv_formatted[len(_skywork_tokenizer.bos_token):]
        
        conv_tokenized = _skywork_tokenizer(conv_formatted, return_tensors="pt").to(_device)
        
        with torch.no_grad():
            score = _skywork_model(**conv_tokenized).logits[0][0].item()

        return {
            'score': score
        }
        
    except Exception as e:
        print(f"Error in Skywork reward computation: {e}")
        return None
    
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
        print(f"Error in Skywork reward computation: {e}")
        return None

def filter_skywork_score(score_result: Dict) -> bool:
    """
    Validate Skywork score results.
    
    Args:
        score_result: Result from compute_skywork_reward
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if score_result is None:
        print('No Skywork result found')
        return False
    
    if not isinstance(score_result, dict):
        print(f"Skywork returned error or incomplete result: {score_result}")
        return False

    score = score_result.get('score')
    
    if score is None:
        print(f"Skywork Score (no score available)")
        return False

    return True

def write_tag_rewards(
    reward_tensor: torch.Tensor,
    batch_idx: int,
    turn_pos: int,
    score: float
):
    """
    Write Skywork reward to the reward tensor.
    
    Args:
        reward_tensor: Tensor to write rewards to [B, seq_len]
        batch_idx: Batch index
        turn_pos: Position in sequence for this turn
        score: Raw score from Skywork model
    """
    normalized_score = torch.sigmoid(torch.tensor(score)).item()
    reward_tensor[batch_idx, turn_pos] = normalized_score