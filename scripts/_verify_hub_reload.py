from transformers import AutoModelForCausalLM
import torch
import tiktoken

model = AutoModelForCausalLM.from_pretrained("msclaw/loom-gpt2-124m", trust_remote_code=True).eval()
enc = tiktoken.get_encoding("gpt2")
ids = torch.tensor([enc.encode("Every effort moves you")])
out = model.generate_simple(ids, max_new_tokens=20)
print(enc.decode(out[0].tolist()))
