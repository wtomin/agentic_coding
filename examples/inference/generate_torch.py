import torch
from transformers import AutoTokenizer, Cohere2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
model = Cohere2ForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-v01", torch_dtype=torch.float16)

# format message with the Command-R chat template
messages = [{"role": "user", "content": "How do plants make energy?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    cache_implementation="static",
)
print(tokenizer.decode(output[0], skip_special_tokens=True))