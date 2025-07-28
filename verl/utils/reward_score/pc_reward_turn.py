import re
import random
import openai
from typing import Dict, List
import torch

# empId = 'buyer-agent-daily'
empId = '468091' # Kunyu's api
api_version = '2024-10-01-preview'

gpt_api_key = 'icbu-azure-buyer-agent-algo'
claude_api_key = 'icbu-aidc-algo'
gemini_api_key = 'icbu-buyer-agent-algo'

qwen_api_key = 'icbu-dashscope-buyer-agent-algo'

# Available models configuration
AVAILABLE_MODELS = {
    'gpt-4': {
        'provider': 'gpt',
        'name': 'gpt-4o',
        'max_tokens': 2048
    },
    'claude-3-sonnet': {
        'provider': 'claude',
        'name': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
        'max_tokens': 1024
    },
    'gpt-4-turbo': {
        'provider': 'gpt',
        'name': 'gpt-4-turbo',
        'max_tokens': 2048
    },
    'gemini-2.5-flash': {
        'provider': 'gemini',
        'name': 'gemini-2.5-flash-preview-04-17',
        'max_tokens': 2048,
        'extra_body': {
            'google': {
                'thinkingConfig': {
                    'includeThoughts': True,
                    'thinkingBudget': 1024
                },
                'thought_tag_marker': 'think'
            }
        }
    },
    'gemini-2.5-pro': {
        'provider': 'gemini',
        'name': 'gemini-2.5-pro-preview-05-06',
        'max_tokens': 2048,
        'extra_body': {
            'google': {
                'thinkingConfig': {
                    'includeThoughts': True # set false to disable showing thinking
                },
                'thought_tag_marker': 'think'
            }
        }
    },
    'qwen-max': {
        'provider': 'qwen',
        'name': 'qwen-max-latest', 
        'max_tokens': 2048
    },
    'qwen3': {
        'provider': 'qwen',
        'name': 'qwen3-235b-a22b',
        'max_tokens': 2048,
        'extra_body': {
            "enable_thinking": False
            }
    }
}

def list_available_models() -> List[str]:
    """Return list of available models"""
    return list(AVAILABLE_MODELS.keys())


def init_clients():
    """Initialize API clients for different providers"""
    gpt_client = openai.AzureOpenAI(
        default_headers={"empId": empId},
        api_key=gpt_api_key,
        api_version=api_version,
        azure_endpoint="https://iai.alibaba-inc.com/azure"
    )

    claude_client = openai.OpenAI(
        default_headers={'empId': empId, 'iai-tag': 'accio'},
        api_key=claude_api_key,
        base_url='https://iai.alibaba-inc.com/aidc/v1',
    )
    
    gemini_client = openai.OpenAI(
        default_headers={'empId': empId},
        api_key=gemini_api_key,
        base_url='https://iai.alibaba-inc.com/aidc/v1'
    )

    qwen_client = openai.OpenAI(
        default_headers={'empId': empId},
        api_key=qwen_api_key,
        base_url='https://iai.alibaba-inc.com/dashscope'
    )

    return {'gpt': gpt_client, 'claude': claude_client, 'gemini': gemini_client, 'qwen': qwen_client}


def get_completion(model_id: str, messages: List[Dict[str, str]]) -> str:
    """
    Get completion from selected model
    
    Args:
        model_id: ID of the model to use (from AVAILABLE_MODELS)
        messages: List of message dictionaries
        
    Returns:
        Model response content
    """
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model ID. Available models: {list_available_models()}")
    
    model_config = AVAILABLE_MODELS[model_id]
    clients = init_clients()
    client = clients[model_config['provider']]
    
    # Prepare request parameters
    params = {
        'model': model_config['name'],
        'messages': messages,
        'max_tokens': model_config['max_tokens']
    }
    
    # Add extra_body for Gemini
    if model_config['provider'] == 'gemini' and 'extra_body' in model_config:
        params['extra_body'] = model_config['extra_body']
    elif model_config['name'].startswith('qwen3'):
        params['extra_body'] = model_config['extra_body']
    
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def build_messages(query_str, solution_str):
    """
    Build messages for LLM as a judge.
    
    Args:
        data: 
    Returns:
        message
    """

    # NOTE dont encourage model to use <search> or <think> that is no need
    sys_prompt = """ 
    You are a strict and skilled evaluator. 
    Given **Query**, and **Response** pair, you should only evaluate the **Response**.
    For evaluation, generate reasonable principles and score each principle individually. 

    [Output Format Requirements]
    Respond in exactly two lines:
        1. Analysis: Explain the reasoning and individual scores for each principle.
        2. Scores: <final_score>SCORE,MAX_SCORE</final_score>. e.g. <final_score>4,6</final_score>
    """.strip()

    user_fmt = f"""
    [Conversation Context]
    **Query**: {query_str}
    **Response**: {solution_str}
    """.strip()
    
    return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_fmt},
    ]

def extract_score(judge_str):
    """Extract the equation from the solution string."""

    final_score_pattern = r'<final_score>(.*?)</final_score>'
    final_score_match = re.finditer(final_score_pattern, judge_str, re.DOTALL)
    final_score_match = list(final_score_match)
    if len(final_score_match) == 0:
        return None
    final_score_str = final_score_match[-1].group(1).strip()

    try:
        score, max_score = final_score_str.split(',')
        score = int(score)
        max_score = int(max_score)
    except Exception as e:
        print(f"Error in extracting score: {e}")
        return None

    return {
        'max_score': max_score,
        'score': score
    }


def compute_score_llm_judge(query_str, solution_str, model_id):
    """The scoring function for LLM-based self-principle judgement. 
       Given a query and a solution, the LLM will judge the solution and return the score.

    Args:
        solution_str: the solution text
        max_score: max score returned by LLM
        
    Returns:
        tuple: (search_score, think_score, final_score, max_score_extracted, response, validation_result)
    """

    # return final score by LLM
    messages = build_messages(query_str, solution_str)
    
    try:
        response = get_completion(model_id, messages)
        #print("\nResponse:", response)
    except Exception as e:
        print('Calling API failure')
        print(f"Error: {str(e)}")
        return None

    # extract score from response
    score_dict = extract_score(response)
    if score_dict is None:
        print('Error in score extraction')
        return None
    
    score = score_dict['score']
    max_score_extracted = score_dict['max_score']

    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print('In processing PRM')
        print(f"Query: {query_str}")
        print(f"Solution string: {solution_str}")
        print(f"LLM judgement: {response}")
        print(f"Score: {score}")
        print(f"Max score extracted: {max_score_extracted}")
        print(f"--------------------------------")

    return score_dict


def write_tag_rewards(
    reward_tensor: torch.Tensor,      # shape = [B, seq_len]
    batch_idx: int,                   # 当前样本在 batch 里的下标
    turn_pos: int,
    score: int,
    max_score_extracted: int,     # max score extracted from LLM judge
):

    reward_tensor[batch_idx, turn_pos] = score/max_score_extracted  # normalize PRM to [0, 1]


def filter_llm_judge_score(score_llm_result):
        """
        Process LLM judge score results and validate them before applying rewards.
        
        Args:
            score_llm_result: Result from compute_score_llm_judge
            response_str: Response string for tag counting validation
            tokenizer: HF tokenizer
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if score_llm_result is None:
            print('No LLM result found')
            return False
        
        # Handle non-dictionary results (error cases)
        if not isinstance(score_llm_result, dict):
            print(f"LLM judge returned error or incomplete result: {score_llm_result}")
            print("NO prm added in this case")
            return False

        max_score_extracted = score_llm_result.get('max_score')
        score = score_llm_result.get('score')
        
        # Check if max_score_extracted is valid
        if max_score_extracted is None or max_score_extracted <= 0:
            print(f"LLM Judge Score (no max score available)")
            return False

        if score is None or score < 0:
            print(f"LLM Judge Score (score < 0)")
            return False

        # Check if sum of scores exceeds max score
        if score > max_score_extracted:
            print(f"LLM Judge Score (sumed tags score > max score extracted)")
            return False

        # success
        return True
