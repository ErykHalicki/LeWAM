# LeWAM v0.3 Architecture

although trained for many less steps than smolvla and other vlas and WAMs, v0.2 failed to learn how to complete a simple single task, with no sign of improvement 
also video prediction loss completely plateaued very early on in training, with the model predicting the last context frame over and over (complete mode collapse)

i beleive this may be due to hte weak language conditioning, since i limited hte model to only 4 VLM layers, projected into a single d_model token (for each token in the sequence)

V0.3 focuses on inherting more priors from the VLM, as well as increasing model size, essentially copying smolVLAs design, but with the added VJEPA2 embeddings as context, and an auxilery loss

Additionally, the approach of v0.2 was too hasty; starting with pretraining before verifying that the model is capable of learning a single task was a mistake.
v0.3 will go from the ground up, starting with a single-task and no video prediction objective to verify that the model works, before proceding to single-task + video prediction, and finally pretraining + multi task fine tuning (with and without video prediction for ablation)

- LeWAM v0.3 
    - use 256 crop following VJEPA2-AC
    - match the DiT depth with the truncated VLM depth (16 DiT layers = 16 VLM layers)
    - maybe try a ViT-L if video prediction performance is still weak (300M)

    - many more VLM layers, different attnetion scheme
        - 12+ layers of smolvlm2, copying smolvla. one layer per DiT layer 
            - start with 256M for single task, if it works, up to 500M for better capacity)
        - get rid of the vlm projection. instead, use smolVLA attention approach of running the VLM and diffusion head in parrallel
        - DiT layer N cross attends to VLM latent at layer N
        - this reduces the need to learn a single unified representation based on only 4 VLM layers, and should allow us to inherit more of the VLMs understanding
    
    - keep vjepa2.1 ViTB as the video encoder
        - use last layer output as context (and ground truth in the video prediction case)
        - keep the vjepa2.1 context prepended to the action sequence as if it was an action (maintain current architecture)
        - this way we are just changing how the model interacts with the VLM hidden states.

    - reduce sequence length, less video context and prediction 
        - maybe even reduce to 3FPS (vjepa2 was trained on 4fps so its not a stretch)
        - ~2.67 sec context = 8 frames = 4 tublets * 256 tokens = 1024 tokens * 768dim = ~1.6Mb context (fp16)
        - ~2.67 sec horizon = 8 frames = 4 tublets * 256 tokens = 1024 tokens * 768dim = ~1.6Mb future (fp16)
            - 2.67 sec horizon = 80 actions @ 30fps = 80 * 768 = ~0.1MB

    - start first experiment with NO video prediction objective
        - then add the video latent prediction objective with the exact same dataset and compare performance (once the task is successfully completable)
        - Collect a CLEAN single task dataset. 5 specific locations, 10 slow demonstrations per location.
