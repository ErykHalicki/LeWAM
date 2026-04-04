import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText

from wam.models.common import make_mlp


class VLMEncoder(nn.Module):
    """
    Extracts conditioning tokens from the first `num_layers` layers of SmolVLM2.

    Text is tokenized and embedded; optional images are encoded through the VLM
    vision encoder + connector and prepended as visual prefix tokens before the
    language model layers. The final hidden state is projected to `model_dim`.

    forward(texts, images):
        texts:  list[str]
        images: (B, C, H, W) float/uint8 tensor or None
        → (B, seq_len, model_dim)
    """

    img_mean: torch.Tensor
    img_std:  torch.Tensor

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        num_layers: int | None = None,
        model_dim: int = 512,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)

        full_model = AutoModelForImageTextToText.from_pretrained(model_id)
        vlm = full_model.model
        lm  = vlm.text_model

        total_layers    = len(lm.layers)
        self.num_layers = num_layers if num_layers is not None else total_layers // 2

        # Truncate to num_layers so lm.forward only runs those layers
        lm.layers = nn.ModuleList(list(lm.layers)[: self.num_layers])

        self.vision_model = vlm.vision_model
        self.connector    = vlm.connector
        self.lm_encoder   = lm

        vlm_hidden_dim = lm.config.hidden_size
        self.proj = make_mlp(vlm_hidden_dim * self.num_layers, model_dim, model_dim)

        for module in (self.vision_model, self.connector, self.lm_encoder):
            for p in module.parameters():
                p.requires_grad_(False)

        # Image preprocessing params extracted from the processor so we can
        # preprocess on GPU without going through numpy.
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
        return next(self.proj.parameters()).device

    @property
    def dtype(self):
        return next(self.proj.parameters()).dtype

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
            tokens:       (B, seq_len, model_dim)
            padding_mask: (B, seq_len) bool, True = padding (ignore in cross-attention)
        """
        inputs    = self.processor(text=texts, return_tensors="pt", padding=True)
        input_ids    = inputs["input_ids"].to(self.device)
        attn_mask    = inputs["attention_mask"].to(self.device)  # 1=real, 0=pad

        with torch.no_grad():
            inputs_embeds = self.lm_encoder.embed_tokens(input_ids)

            if images is not None:
                vision_dtype   = next(self.vision_model.parameters()).dtype
                pixel_values   = self._preprocess_images(images.to(self.device)).to(vision_dtype)
                image_features = self.vision_model(pixel_values=pixel_values).last_hidden_state
                image_features = self.connector(image_features).to(inputs_embeds.dtype)
                B, N_img, _ = image_features.shape
                inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)
                attn_mask     = torch.cat([attn_mask.new_ones(B, N_img), attn_mask], dim=1)

            out = self.lm_encoder(inputs_embeds=inputs_embeds, use_cache=False, output_hidden_states=True)
            hidden = torch.cat(out.hidden_states[1:], dim=-1)

        tokens       = self.proj(hidden.to(self.dtype))
        padding_mask = attn_mask == 0  # True = ignore
        return tokens, padding_mask
