from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
import re

local_path = copy_local_path_from_hdfs('Qwen/Qwen2.5-3B-Instruct')
tokenizer = hf_tokenizer(local_path)

# ① 得到 token-id 序列
text = "<think>\nOkay, here's how I'm going to approach this evaluation. The user wants me to assess the given reasoning, which includes a series of 'think' and 'search' actions. My job is to dissect each of these, assigning scores based on predefined criteria.\n\nFirst, I need to meticulously evaluate each `<think>` tag.  For the first one, I'll check if the reasoning appropriately initiates a `<search>` given the lack of initial information - in this case the unknown singer. Then, I will consider if it correctly identifies that a search is needed, it gets a point. Then, I assess whether the reasoning is sound and focused on the relevant information.\n\nNext, I'll move on to the `<search>` tags. For each of these, I'll evaluate the search itself, considering whether it's justified given the current information, the entities it identifies, and how clearly and concisely it's formulated.\n\nFor the second and third `<think>` tags, I'll repeat the process, checking if another search is needed based on the information gained, and if so, if it correctly extracts the relevant information (like the singer and birthplace) and if the reasoning aligns with the objective of the search.\n\nFinally, I'll get to the fourth `<think>` tag, at which point I'm looking at whether the system rightly decides *not* to perform a search. If it does not need a search, then it gets a score of 0 for the first criteria, then I'll again check the information extraction and reasoning.\n\nI will then calculate the maximum possible score for all the `<think>` tags (a maximum of 9 points), and the maximum possible score for all the `<search>` tags (another maximum of 9 points). Finally, I will combine these two maximum scores to get the overall maximum score (a total of 18 points) and apply that to the provided solution.\n \n</think>\n\nAnalysis: <think> 1:\n            T1: Yes — correctly decides to perform a <search> due to insufficient information (the singer's identity).\n            → Score: 1/1\n\n            <search> 1:\n            Justified: There is no prior information about the fragrance or singer.\n            S1: Yes — contains \"Curious fragrance\".\n            S2: Yes — clearly phrased.\n            S3: Yes — concise.\n            → Score: 3/3\n\n            <think> 2:\n            T1: Yes — correctly decides to perform a <search> as the specific city and state of birth are still unknown.\n            T2: Yes — correctly extracts \"Britney Spears\" as the singer associated with \"Curious\".\n            T3: Yes — the reasoning is consistent with the query's goal of finding the birthplace.\n            → Score: 3/3\n\n            <search> 2:\n            Justified: Needs to find Britney Spears' birthplace.\n            S1: Yes — contains \"Britney Spears\".\n            S2: Yes — clearly phrased.\n            S3: Yes — concise.\n            → Score: 3/3\n\n            <think> 3:\n            T1: Yes — correctly decides to perform a <search> to confirm the location of \"McComb, Mississippi\" as requested by the query's format (city and state).\n            T2: Yes — correctly extracts \"McComb, Mississippi\" as Britney Spears' birthplace.\n            T3: Yes — the reasoning is consistent with finding the exact city and state for the final answer.\n            → Score: 3/3\n\n            <search> 3:\n            Justified: Needs to confirm the city and state of \"McComb, Mississippi\".\n            S1: Yes — contains \"McComb, Mississippi\".\n            S2: Yes — clearly phrased.\n            S3: Yes — concise.\n            → Score: 3/3\n\n            <think> 4:\n            T1: Not applicable, as it correctly determines that no further search is needed to answer the question, as the required information (city and state) has been obtained.\n            T2: Yes — correctly extracts \"McComb, Mississippi\" as the final answer.\n            T3: Yes — the reasoning is sound and confirms the availability of all necessary information for the answer.\n            → Score: 2/2\n        max_score=1+3+3+3+3+3+2=18\nScores: <final_score><search>3,3,3</search><think>1,3,3,2</think><max_score>18</max_score></final_score>"
ids = tokenizer.encode(text, add_special_tokens=False)
print("token ids :", ids)

# ② 把 id 反查成词表里的 token 形态
print("tokens    :", tokenizer.convert_ids_to_tokens(ids))

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

search_pos = find_paired_tag_positions(text, 'search', tokenizer)
think_pos = find_paired_tag_positions(text, 'think', tokenizer)
print(search_pos)
print(think_pos)

for pos in search_pos:
    print(tokenizer.convert_ids_to_tokens(ids[pos]))

for pos in think_pos:
    print(tokenizer.convert_ids_to_tokens(ids[pos]))