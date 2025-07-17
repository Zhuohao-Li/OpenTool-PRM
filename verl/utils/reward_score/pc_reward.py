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
        'name': 'qwen-max-latest',   # TODO: check the model config
        'max_tokens': 2048
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
    You are a strict and skilled evaluator. You should evaluate given responses based on the given criteria.
    Given every occurrence of **<search>** and **<think>**, score it individually based on the corresponding criteria below. 
    The number of scores must match exactly the number of <search> and <think> tags in the <Response to be Scored>.
    Do not skip any. Do not combine multiple ones into a single score. Number of output scores must exactly match number of tags <search> and <think>.

    [Think Evaluation]
        T1. Does the <think> correctly decide to perform a <search> due to insufficient information? If yes, assign 1 point. If no, assign 0 point.
        If <information> was just provided, evaluate the following (1 point for each):
            T2. Fact Extraction: Does it correctly extract key facts (who, what, when) from the given information?
            T3. Relevant Reasoning: Is the reasoning consistent with the original query and focused on solving it?

    [Search Evaluation]
        Only evaluate a <search> if it is correctly triggered, i.e., if no sufficient information is already available. If not, assign 0 to the entire search occurrence.
        If the search is justified, score the following (1 point for each):
            S1. Key Entities: Does the query contain the necessary named entities (e.g., people, show titles)?
            S2. Clarity: Is the query free from ambiguity and clearly phrased?
            S3. Conciseness: Is the query free from unnecessary or redundant phrasing?

    [Output Format Requirements]
    Respond in exactly two lines:
        1. Analysis: Explain the reasoning and individual scores for each <search> and <think>.
        2. Scores: Use the format: <final_score><think>SCORE1,SCORE2...</think><search>SCORE1,SCORE2...</search><max_score>MAX_SCORE_CAN_BE_JUDGED</max_score></final_score>. Example: <final_score><think>1,3</think><search>3,2</search><max_score>8</max_score></final_score>

    **Example**:
    [Conversation Context]
    <Query>: Who was the winner of the season of The Rap Game where a contestant named Lady Leshurr as one of her influences?
    <Response to be Scored>: <think> I need to find the winner of the season of The Rap Game where a contestant named Lady Leshurr was one of her influences. I’ll search for it. </think> <search> who was the winner of the season of The Rap Game where a contestant named Lady Leshurr as one of her influences </search> <information> Doc 1(Title: ”Lady Leshurr”) Act. Leshurr came out as pansexual in September 2018. Leshurr has named Ashley Walters, Lisa Maffia, Missy Elliott, and Lil Wayne as her primary influences. Lady Leshurr Melesha Katrina O’Garro (born 15 December 1989) known professionally as Lady Leshurr (), is a British grime and hip hop rapper, singer, and producer from Kingshurst, Solihull, UK. She is best known for her ””Queen’s Speech”” series of freestyles, the fourth of which went viral in 2016. ””The Rap Game”” season 2 competitor Nia Kay stated that Leshurr was one of herinfluencesduetoheruniquenessandrapskill.Hersubsequentfreestyle,... </information> <think> I found out that Lady Leshurr was one of the influences for Nia Kay in The Rap Game season 2. Now I need to find the winner of that season. </think> <search> winner of The Rap Game season 2 </search> <information> Doc 1(Title: ”Chandra Davis”) Chandra Davis Chandra Davis (born January 2, 1978 in Detroit, Michigan), a.k.a. London Charles and Deelishis, is an American model, and former reality show contestant. She is best known as the winner of VH1’s hit reality TV show ””Flavor of Love 2””. Davis was a contestant on the second season of VH1’s Flavor of Love (aged 29). She went on to win the competition when Flavor Flav chose her over Tiffany ’New York’ Pollard, who returned after previously named the runner-up on the show’s first season. After FOL, Davis announcedsheiscurrentlynotdatingFlav,viaaVH1special,... </information> <think> I found out that the winner of The Rap Game season 2 is London Charles (Chandra Davis). Now I can provide the answer. </think> <answer> London Charles </answer>

    <Output>:
    Analysis: <think> 1:
            T1: Yes — correctly decides to perform a <search> due to no info yet.
            → Score: 1/1

            <search> 1:
            Justified: There is no prior information.
            S1: Yes — includes “The Rap Game” and “Lady Leshurr”.
            S2: Yes — intent is clear.
            S3: No — query has redundancy.
            → Score: 2/3

            <think> 2:
            T1: Yes — correctly decides to perform a <search> due to no info yet.
            T2: Yes — correctly extracts that Nia Kay (season 2) was influenced by Lady Leshurr.
            T3: Yes — reasoning is focused on query and logically moves to next step.
            → Score: 3/3

            <search> 2:
            Justified after identifying the season.
            S1: Yes — “The Rap Game season 2”.
            S2: Yes — unambiguous.
            S3: Yes — concise.
            → Score: 3/3

            <think> 3:
            T1: Not applicable (no new search triggered).
            T2: No — misinterprets "Flavor of Love 2" as relevant.
            T3: No — reasoning is off-topic (wrong show).
            → Score: 0/2
        max_score=1+3+3+3+2=12
    Scores: <final_score><think>1,2,0</think><search>2,3</search><max_score>12<max_score></final_score>
    """.strip()

    user_fmt = f"""
    [Conversation Context]
    Query: {query_str}
    Response to be Scored: {solution_str}
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

    search_score_pattern = r'<search>(.*?)</search>'
    think_score_pattern = r'<think>(.*?)</think>'
    max_score_pattern = r'<max_score>(.*?)</max_score>'

    search_match = re.finditer(search_score_pattern, final_score_str, re.DOTALL)
    search_match = list(search_match)
    think_match = re.finditer(think_score_pattern, final_score_str, re.DOTALL)
    think_match = list(think_match)
    max_score_match = re.finditer(max_score_pattern, final_score_str, re.DOTALL)
    max_score_match = list(max_score_match)

    if len(search_match) == 0 or len(think_match) == 0:   # TODO: do we need to score incomplete think?
        return None

    search_scores = [int(x.strip()) for x in search_match[-1].group(1).strip().split(',') if x.strip()]
    think_scores = [int(x.strip()) for x in think_match[-1].group(1).strip().split(',') if x.strip()]
    
    # Extract max_score if available
    max_score = None
    if len(max_score_match) > 0:
        max_score = int(max_score_match[-1].group(1).strip())
    
    # Calculate final score (sum of all search and think scores)
    final_score = sum(search_scores) + sum(think_scores)

    return {
        'search_score': search_scores,
        'think_score': think_scores,
        'max_score': max_score,
        'final_score': final_score
    }


def compute_score_llm_judge(query_str, solution_str, model_id, max_score=1.):
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
        print(f"Error: {str(e)}")  # TODO possible failed case check, due to IO failure
        return [], [], None, None, None, None

    # extract score from response
    score_dict = extract_score(response)
    if score_dict is None:
        return [], [], None, None, response, None

    search_score = score_dict['search_score']  # NOTE: a list
    think_score = score_dict['think_score']
    max_score_extracted = score_dict['max_score']

    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Query: {query_str}")
        print(f"Solution string: {solution_str}")
        print(f"Max score: {max_score}")
        print(f"LLM judgement: {response}")
        print(f"Search score: {search_score}")
        print(f"Think score: {think_score}")

        print(f"Max score extracted: {max_score_extracted}")
    
    return {
        'search_score': search_score,
        'think_score': think_score,
        'max_score_extracted': max_score_extracted,
        'response': response
    }

def find_paired_tag_positions(text, tag_name, tokenizer):
    """Find paired tags and return the last token position of closing tags"""
    open_pattern = f'<{tag_name}>'
    close_pattern = f'</{tag_name}>'
    
    # Find all opening and closing tag positions
    open_spans = [m.span() for m in re.finditer(re.escape(open_pattern), text)]
    close_spans = [m.span() for m in re.finditer(re.escape(close_pattern), text)]
    
    # Match opening and closing tags
    paired_close_positions = []
    open_idx = 0
    close_idx = 0
    
    while open_idx < len(open_spans) and close_idx < len(close_spans):
        open_start = open_spans[open_idx][0]
        close_start = close_spans[close_idx][0]
        
        # Find the first closing tag after this opening tag
        if close_start > open_start:
            # Found a valid pair
            paired_close_positions.append(close_spans[close_idx][1])  # end position of closing tag
            open_idx += 1
            close_idx += 1
        else:
            # Skip this closing tag (it's before the opening tag)
            close_idx += 1
    
    # Convert character positions to token positions
    if not paired_close_positions:
        return []
        
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offset_map = enc["offset_mapping"]
    char2tok = {}
    for tid, (s, e) in enumerate(offset_map):
        for c in range(s, e):
            char2tok[c] = tid
    
    # Return the last token position of each closing tag
    return [char2tok[end-1] for end in paired_close_positions if end-1 in char2tok]



def write_tag_rewards(
    reward_tensor: torch.Tensor,      # shape = [B, seq_len]
    batch_idx: int,                   # 当前样本在 batch 里的下标
    response_str: str, 
    search_scores: list[float],       # 每个 <search> 的分
    think_scores:  list[float],       # 每个 <think>  的分
    tokenizer,                        # HF tokenizer
    max_score_extracted: int,     # max score extracted from LLM judge
):
    """write score to reward_tensor"""
    pos_search = find_paired_tag_positions(response_str, 'search', tokenizer)
    pos_think = find_paired_tag_positions(response_str, 'think', tokenizer)

    # If not passing this assertion, it means the string match is not correct
    assert len(pos_search) == len(search_scores), \
        f"search tags number {len(pos_search)} ≠ search scores number {len(search_scores)}"
    assert len(pos_think) == len(think_scores),   \
        f"think tags number {len(pos_think)} ≠ think scores number {len(think_scores)}"

    def assign(pos_list, score_list):
        for p, s in zip(pos_list, score_list):
            reward_tensor[batch_idx, p] = s/max_score_extracted  # normalize PRM to [0, 1]

    assign(pos_search, search_scores)
    assign(pos_think,  think_scores)


def filter_llm_judge_score(score_llm_result, response_str):
        """
        Process LLM judge score results and validate them before applying rewards.
        
        Args:
            score_llm_result: Result from compute_score_llm_judge
            response_str: Response string for tag counting validation
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if score_llm_result is None:
            return False
        
        # Handle non-dictionary results (error cases)
        if not isinstance(score_llm_result, dict):
            print(f"LLM judge returned error or incomplete result: {score_llm_result}")
            print("NO prm added in this case")
            return False

        max_score_extracted = score_llm_result.get('max_score_extracted')
        search_scores = score_llm_result.get('search_score', [])
        think_scores = score_llm_result.get('think_score', [])
        sumed_tags_score = sum(search_scores) + sum(think_scores)
        
        # Check if max_score_extracted is valid
        if max_score_extracted is None or max_score_extracted <= 0:
            print(f"LLM Judge Score (no max score available)")
            return False
            
        # Check if sum of scores exceeds max score
        if sumed_tags_score > max_score_extracted:
            print(f"LLM Judge Score (sumed tags score > max score extracted)")
            return False
        
        # Validate that the number of tags matches the number of scores
        search_pattern = r'<search>(.*?)</search>'
        think_pattern = r'<think>(.*?)</think>'
        
        search_matches = re.findall(search_pattern, response_str, re.DOTALL)
        think_matches = re.findall(think_pattern, response_str, re.DOTALL)
        
        search_count = len(search_matches)
        think_count = len(think_matches)
        
        if len(search_scores) != search_count:
            print(f"WARNING: Search tag count mismatch! Tags: {search_count}, Scores: {len(search_scores)}")
            return False
            
        if len(think_scores) != think_count:
            print(f"WARNING: Think tag count mismatch! Tags: {think_count}, Scores: {len(think_scores)}")
            return False

        # success
        return True
