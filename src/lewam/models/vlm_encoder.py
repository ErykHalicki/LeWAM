import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from transformers.masking_utils import create_causal_mask


class VLMEncoder(nn.Module):
    """
    Per-layer conditioning from the first `num_layers` layers of SmolVLM2.

    Text is tokenized and embedded; optional images are encoded through the VLM
    vision encoder + connector and prepended as visual prefix tokens before the
    language model layers. Each layer's hidden state is returned separately for
    SmolVLA-style per-layer cross-attention in the DiT.

    forward(texts, images):
        texts:  list[str]
        images: (B, C, H, W) float/uint8 tensor or None
        → (list[(B, seq_len, vlm_hidden_dim)], (B, seq_len))
    """

    img_mean: torch.Tensor
    img_std:  torch.Tensor

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        num_layers: int | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)

        if pretrained:
            full_model = AutoModelForImageTextToText.from_pretrained(model_id)
        else:
            cfg = AutoConfig.from_pretrained(model_id)
            full_model = AutoModelForImageTextToText.from_config(cfg)
        vlm = full_model.model
        lm  = vlm.text_model

        total_layers    = len(lm.layers)
        self.num_layers = num_layers if num_layers is not None else total_layers // 2

        self.vlm_hidden_dim = lm.config.hidden_size
        self.lm_config = lm.config

        self.vision_model = vlm.vision_model
        self.connector    = vlm.connector
        self.lm_embed     = lm.embed_tokens
        self.lm_layers    = nn.ModuleList(list(lm.layers)[:self.num_layers])
        self.rotary_emb   = lm.rotary_emb

        for module in (self.vision_model, self.connector, self.lm_embed,
                       self.lm_layers, self.rotary_emb):
            for p in module.parameters():
                p.requires_grad_(False)

        ip   = self.processor.image_processor
        size = ip.size
        try:
            self.img_h = int(size.get("height", size.get("shortest_edge", 384)))
            self.img_w = int(size.get("width", self.img_h))
        except (AttributeError, TypeError):
            self.img_h = self.img_w = int(size)

        self.register_buffer("img_mean", torch.tensor(ip.image_mean).view(1, 3, 1, 1))
        self.register_buffer("img_std",  torch.tensor(ip.image_std).view(1, 3, 1, 1))

    @property
    def device(self):
        return next(self.lm_embed.parameters()).device

    @property
    def dtype(self):
        return next(self.lm_embed.parameters()).dtype

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """GPU-side image preprocessing: resize + normalize. (B,C,H,W) → (B,C,H,W)"""
        if images.dtype == torch.uint8:
            x = images.float() / 255.0
        else:
            x = images.float().clamp(0.0, 1.0)
        x = F.interpolate(x, size=(self.img_h, self.img_w), mode="bilinear", align_corners=False)
        return (x - self.img_mean) / self.img_std

    def forward(self, texts: list[str], images: torch.Tensor | None = None):
        """
        Returns:
            per_layer:    list of num_layers tensors, each (B, seq_len, vlm_hidden_dim)
            padding_mask: (B, seq_len) bool, True = padding (ignore in cross-attention)
        """
        inputs    = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids    = inputs["input_ids"].to(self.device)
        attn_mask    = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            hidden = self.lm_embed(input_ids)

            if images is not None:
                vision_dtype   = next(self.vision_model.parameters()).dtype
                pixel_values   = self._preprocess_images(images.to(self.device)).to(vision_dtype)
                image_features = self.vision_model(pixel_values=pixel_values).last_hidden_state
                image_features = self.connector(image_features).to(hidden.dtype)
                B, N_img, _ = image_features.shape
                hidden    = torch.cat([image_features, hidden], dim=1)
                attn_mask = torch.cat([attn_mask.new_ones(B, N_img), attn_mask], dim=1)

            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            causal_mask = create_causal_mask(
                config=self.lm_config,
                inputs_embeds=hidden,
                attention_mask=attn_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            position_embeddings = self.rotary_emb(hidden, position_ids=position_ids)

            per_layer = []
            for layer in self.lm_layers:
                hidden = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                per_layer.append(hidden)

        padding_mask = attn_mask == 0
        return per_layer, padding_mask
