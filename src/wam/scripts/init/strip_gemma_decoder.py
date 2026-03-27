from wam.models.lewam import load_t5gemma_encoder

t5gemma_checkpoint = 'weights/t5gemma-s-s-prefixlm'

encoder = load_t5gemma_encoder(path=t5gemma_checkpoint)

encoder.backbone.save_pretrained(t5gemma_checkpoint, use_safetensors=True)

