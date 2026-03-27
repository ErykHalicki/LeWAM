
import torch
from transformers import AutoTokenizer, T5GemmaEncoderModel

model_id = "google/t5gemma-s-s-prefixlm"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = T5GemmaEncoderModel.from_pretrained(
  model_id,
  torch_dtype=torch.bfloat16,
  device_map="auto",
  is_encoder_decoder=False,
)
model.eval()

text = ["Hello, world!", "This is a test sentence."]
inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

token_embeddings = outputs.last_hidden_state
print("Token embeddings shape:", token_embeddings.shape)
