import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda:0"
model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."

# 单条推理（for循环100次）
conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
    conv_formatted = conv_formatted[len(tokenizer.bos_token):]
conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)

start = time.time()
for _ in range(100):
    with torch.no_grad():
        score = rm(**conv_tokenized).logits[0][0].item()
end = time.time()
print(f"Single (repeat 100) total time: {end - start:.3f}s")

# batch 推理（一次性100条）
convs = [conv for _ in range(100)]
conv_formatted_batch = [tokenizer.apply_chat_template(c, tokenize=False) for c in convs]
if tokenizer.bos_token is not None:
    conv_formatted_batch = [
        s[len(tokenizer.bos_token):] if s.startswith(tokenizer.bos_token) else s
        for s in conv_formatted_batch
    ]
batch = tokenizer(conv_formatted_batch, return_tensors="pt", padding=True, truncation=True).to(device)

start = time.time()
with torch.no_grad():
    logits = rm(**batch).logits.squeeze(-1)  # shape: [100]
    scores = logits.tolist()
end = time.time()
print(f"Batch (100) total time: {end - start:.3f}s")
print(f"First 5 batch scores: {scores[:5]}")