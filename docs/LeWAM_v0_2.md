# LeWAM v0.2 Architecture

attempt #2, moving to a more precedented and simple architecture. Essentially LeWorldModel + DreamZero + SmolVLA (kind of)
main updates:
    - removing the IDM and following a dreamzero-style singular model approach (one DiT jointly predicting actions and vieo latents)
    - downscaling the vjepa model outputs using an MLP adaptor with SIGReg regularization to prevent representation collapse
    - Improving language conditioning to use a VLM instead of an LLM, allowing for out of the box vision alignment 
    - Autoregressive masking
    - stiching video frames side by side like dreamzero to simplify arch

DreamZero uses a large pretrained video generation diffusion model, which is expensive because it trains up to 720p (or higher) and attempts to 
generate dense pixel representations. However, the JEPA theory is that if the model works in latent space, training is cheaper since we discard visual information that is 
unimportatnt to the task and hard to predict 
Practically, this means we would generate just video latents, with other latents as ground truth instead of actual images. 
Although the actual DiT in this model is not initialized from a strong model like DreamZero, the pretrained VJEPA2 embeddings were trained with a prediction objective on internet scale data, so they should be a strong warm start for the model, allowing efficient training.

Could also downscale dimension size + finetune the vision representation by adding an MLP head and freezing the actual encoder. Additionally applying SIGReg to the MLP output to ensure that the adaptor doesnt collapse the representation.
In this case, the model is able to update the video encoder output, without destroying the VJEPA2 weights.

## Architecture design
- Reference: https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py
- Need to subsample the dataset from 30fps -> 10fps, following dreamzero
- all resolutions and scales are examples (except the VJEPA2 tubelet size N/2, which is fixed)
- Video path: 
    - video context + future frames (N frames) -> VJEPA2 -> raw video embeddings (N/2 frames) -> mean pool -> pooled frames (N/10 frames) -> 
    - MLP adaptor -> pooled and downscaled video embeddings (N/10 frames, 0.75 x vjepa2 dim) -> DreamZero-style DiT (with context frames as input) -> 
    - pred vid latent vel -> mse_loss(pred_vel, x1-x0) + SIGReg(vid_emb)
        - x1 in this case is the encoded future latents (from before the DreamZero DiT)
        - so the training objective is still JEPA-style, since we are trying to predict our video encoders own latents
- Action path:
    - Action noise + state -> action + state encoder -> DreamZero-style DiT -> action decoder

- Text + Image conditioning path:
    - should follow cosmo predict 2.5 arch (the model mimic video and dreamzero inherit from)
    - replacing the cosmo-reason1 model with SmolVLM2-256M https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
    - text instruction + video context -> VLM -> layer latents -> concat first N latents (chop off remaining layers) -> MLP -> 
    - sequence of language and visual latents in model dim -> DreamZero style DiT (Cross attention layers)

The biggest unknown is if training in latent space will work, but the rest of the architecture is pretty much the same as DreamZero, just with a downscaled VLM encoder and smaller DiT

Target sizes of model:
- Base version for main training experiment
    - 256M VLM, use first half of the layers for conditioning vector -> ~128M params
    - VJEPA2.1 80M, use entire model -> 80M 
    - DiT -> 300M (including all action and state de/encoders and all adaptor MLPs) 
    - total param count: 128M + 80M + 300M -> 500M params

- Small version for overfitting / single task on one dataset
    - 256M VLM, use first quarter of the layers for conditioning vector -> ~64M params
    - VJEPA2.1 80M, use entire model -> 80M 
    - DiT -> 100M (including all action and state de/encoders and all adaptor MLPs) 
    - total param count: 64M + 80M + 100M -> 250M params

Training duration / details:
    - general consensus of papers ive seen so far is 50-200K steps with 32-256 batch size
    - almost everyone uses LR warmup then annealing
    - LR 1e-4 -> 5e-6 over 50k steps
    - 1000 step warmup from 5e-6 -> 1e-4
    - AdamW with default 
    - freeze all pretrained models (VJEPA2.1 80M, smolVLM 256M) 

Implementation references: 
    - https://github.com/huggingface/VLAb/blob/main/src/lerobot/policies/smolvla2/smolvlm_with_expert2.py
    - https://github.com/dreamzero0/dreamzero/blob/main/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py


## Implementation notes:
    - i think its important to detach the future embeddings from the loss calculation
    - so far it looks like the VideoAdaptor needs a higher LR than the rest of hte system to work
        - or maybe just exclude it?

