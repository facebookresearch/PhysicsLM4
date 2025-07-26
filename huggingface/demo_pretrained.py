# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code loads pretrained models from huggingface.co

from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose any of our 16 released models
# model_name = "facebook/PhysicsLM4.2__LlamaCanon-8B-Nemo-1T-lr0.003"
# model_name = "facebook/PhysicsLM4.2__LlamaCanon-1B-Nemo-2T-lr0.005"
model_name = "facebook/PhysicsLM4.2__Llama-3B-Nemo-1T-lr0.003"

# Below is simply a wrapper for either the Llama2 tokenizer (for <=3B models) or Llama3 (for 8B models);
#   alternatively, you can download your own HF llama2/3 tokenizers and use that instead
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

input_text = "Galileo Galilei climbed the Leaning Tower of Pisa to conduct a controlled experiment"
inputs = tokenizer(input_text, return_tensors="pt")

output_ids = model.generate(inputs['input_ids'].cuda(), max_new_tokens=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print('-'*50)
print(output_text)
