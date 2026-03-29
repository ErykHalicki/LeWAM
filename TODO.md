
## Implementation

- arch notes:
    - pi0 and 0.5
    - smolVLA
        - n=50 action chunks
        - 100-step warmup, cosine lr sched 1e-4 -> 2.5e-6
        - AdamW optimizer with β1 = 0.9, β2 = 0.95.
        - 450M params, VLM backbone fully frozen 
            - 100M for action expert
        - 200k steps @ 256 batch size for pre training. 200k steps @ 64 batch size for fine tuning
            - 5s per step on 2x5090 (optimistic) = 280 hrs = $500CAD for pre training
        - CA + SA is good, skipping VLM layers is good (instead of downsizing)
        - Causal attention mask on the DiT improved performance
        - 512x512 image resolution (16x16 patch size -> 32x32 patches -> 1024 tokens)
    - mimic video
        - only had to train for ~35k steps @ 128 batch
        - partially denoised video (tau=0.5-0.8) performed best as input to action head? fully denoised video was not beneficial since its not fully accurate 
        - Vision backbone: SA to full video sequence (clean + noisy latents) -> CA to language instructions -> MLP
        - Action decoder: CA over video latents -> SA over action sequence -> MLP
        - Uses AdaLN with both video and action flow times (tau_v and tau_a) as input
        - DiT on both video and action prediction
            - seperate flow times per head (video and action denoised with different tau)
        - Video LoRA finetune pretraining -> action policy training (frozen backbone)
        - Single task diffusion policy uses 155M params in multi view case
    - Cosmo predict 2.5 (dreamzero and mimic_video generation backbone)
        - concats multiple activations from an LLM / VLM and projects them into their latent dim 
            - essentially taking info from multiple layers of the LLM
            - keeps sequence length the same, but projects them from N*LLM_latent_dim -> video_latent_dim
        - cross attention over the language tokens, self attention over ENTIRE video sequence (future noise + past gt)
        - text encoder is a VLM, so image information enters from 2 streams, both the VLM and the diffusion process
        - generates 93 frames at a time, which is 24 latents with their reduction strategy
        - 320x192 -> 832x480 -> 1280x704 as training goes on, moving between steps once loss plataeus 
        - noise timestep is sampled using a logit-normal distrubtion, not uniform. https://en.wikipedia.org/wiki/Logit-normal_distribution
            - as training went on, they used a "timestep shift" from 1->5 (not sure how they are applying this?, do they mean sigma went up, forcing the distrubtion closer to 1?) 
        - AdamW with 0.9 0.999 with 0.001 weight decay. 3e-5 LR with 2000 iteration warmup and linear LR decay.
        fine tuned for 30k steps @ 256 batch (unknown how long pre training was)
    - dream zero
        - action and video prediction are both part of 1 DiT model so they can attend to each other
            - observations are chunked (K=2) so frames are paired in the attention mask. this appearntly performs better than K=1
            - number of chunks M = 4. so the model predicts 4 chunks, which are then decoded to images and actions repsectively
            - smaller ablation models trained for 50K steps @ 32 batch size (god bless maybe this is tractable)
        - applies acton chunk smoothing to counter high frequency noise from the diffusion process
        - heavily subsamples fps (5 fps for video and 15-30fps for actions)
        - 

    - dreamer v3 (and other latent space models?)
    - LeWorldModel 
        - pretty much just VJEPA2
    - VJEPA2 ML-LLM alignemnt uses early fusion like LLaVA (just append the projected video embeddings with the language embeddings)
        - projection done by a 2 layer MLP

- improve training setup to use accelerate library
    - clean it up. make a new script
- run experiments training the policy

### Optional / low priority
- move vjepa tublet size to leWAm as a constant
- create a decoder for visualizing predicted embeddings
    - can be trained on the embedding - image pairs from just the encoder, and then used to visualize predictor outputs
- add a simple embodiment configuration system
    - or at least add automatic training support for multiple embodiments
- move the precomputation scripts into the training scripts folder from the general scripts
- make the policy compatible with lerobot tools (make a lerobot policy)
    - https://huggingface.co/docs/lerobot/bring_your_own_policies

## Evaluation
- What benchmark to use for testing and comparing to other approaches
    - Libero? 
    - My own suite of comparisons? 
        - eg. also train an ACT, BC, Diffusion policy, smolVLA, etc. (these can all be implemented and trained using lerobot very easily!)
    - for baselines, to be fair, make a BC policy that uses vjepa2 as its visual encoder
- rollout policy on So101 in MuJoCo or Isacc sim 
- Test on real hardware depending on simulated results

## Research
- A key weakness of VJEPA2 is that it requires search at inference time to generate actions 
    - if we instead replace
- update architecture.tex to match real code
    - make architecture diagram for DiT specifically as well
- write up basic motivation in main.tex
- you can look at the TeX source of paper on arxiv for reference on how to make figures for example
- read about other world models referenced in dreamzero and cite them

