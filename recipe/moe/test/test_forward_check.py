import pickle as pkl
import sys

import torch

sys.path.insert(0, "/newcpfs/user/liuyanjiang/Eng/agi-verl/recipe/moe")
from moe_trainer.bitdump import hook_fwd_bwd_to_module
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/cpfs/user/liuyanjiang/hf_models/moe_sft_145b_32k_v7.2.0_CIF_iter_0001589_hf-3layers",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "/cpfs/user/liuyanjiang/hf_models/moe_sft_145b_32k_v7.2.0_CIF_iter_0001589_hf-3layers", trust_remote_code=True
)
model.to(torch.device("cuda"))
model.train()
names = None
dump_path = "/newcpfs/user/liuyanjiang/Eng/agi-verl/recipe/moe/moe_trainer/dump_dir_2"
hook_fwd_bwd_to_module(model, names=names, prefix=f"{dump_path}/")

with open("/newcpfs/user/liuyanjiang/Eng/agi-verl/recipe/moe/test/inputs.pkl", "rb") as f:
    inputs = pkl.load(f)

for k, v in inputs.items():
    inputs[k] = v.to(model.device)
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(**inputs)

# out = model(**inputs)

logits = output["logits"].detach().cpu().numpy()
print(logits)
# with open('./patch.pkl', 'wb')as f:
#     pkl.dump(logits, f)
