# %%
import torch
from transformers import AutoModel, AutoTokenizer

# %%
# model_name = "GSAI-ML/LLaDA-8B-Base"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
# ).eval()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
# %%


tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True, torch_dtype=torch.bfloat16
)
# %%
model = model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# %%
text = "Large Language Diffusion Models are exciting."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
# %%
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
last_hidden = outputs.last_hidden_state.squeeze(0)  # shape [seq_len, hidden_dim]
mask = attention_mask.squeeze(0).unsqueeze(-1)  # shape [seq_len, 1]


# %%
# zero‑out padding positions before pooling
sum_embeddings = (last_hidden * mask).sum(dim=0)
token_counts = mask.sum()  # number of non‑padding tokens
sentence_embedding = sum_embeddings / token_counts
