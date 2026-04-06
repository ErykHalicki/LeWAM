## Implementation

- LeWAM v0.3 
    - use 256 crop following VJEPA2-AC
    - match the DiT depth with the truncated VLM depth (16 DiT layers = 16 VLM layers)
    - maybe try a ViT-L if video prediction performance is still weak (300M)

    - get rid of GEGLU activation, just use GELU for simplicity

    - move video and action projection BEFORE noise addition, following dreamzero
        - add noise to d_model not d_video and d_action seperately
    
    - more VLM layers, same VLM size
        - 16 layers of 500M smolvlm2, copying smolvla. one layer per DiT layer 
        - get rid of the vlm projection. instead, use smolVLA attention approach of running the VLM and diffusion head in parrallel
        - DiT layer N cross attends to VLM latent at layer N
        - this reduces the need to learn a single unified representation based on only 4 VLM layers, and should allow us to inherit more of the VLMs understanding
    
    - keep vjepa2.1 ViTB as the video encoder
        - use last layer output as context (and ground truth in the video prediction case)
        - keep the vjepa2.1 context prepended to the action sequence as if it was an action (maintain current architecture)
        - this way we are essentially just changing how the model interacts with the VLM hidden states.

    - reduce sequence length, less video context and prediction 
        - maybe even reduce to 3FPS (vjepa2 was trained on 4fps so its not a stretch)
        - ~3.3 sec context = 10 frames = 5 tublets * 256 tokens per tublet = 1280 tokens * 768dim = ~2Mb context (fp16)
        - 2 sec horizon = 6 frames = 3 tublets * 256 tpt = 768 tokens * 768 dim = ~0.6Mb future (fp16)
            - 2 sec horizon = 60 actions @ 30fps = 60 * 768 

    - start first experiment with NO video prediction objective
        - then add the video latent prediction objective with the exact same dataset and compare performance (once the task is successfully completable)
        - Collect a CLEAN single task dataset. 5 specific locations, 10 slow demonstrations per location.

--- 

- write up experiment design in paper
    - stick to what smolvla did 

- reach out to mees for feedback

- specify that we use a relative joint space delta action space in the paper

- check euler wait times
    - plan out a potential training run depending on rollout results
    - run fine tuning experiments on euler?

- run the teacher forcing ablation from mimic video 
    - if we provide ground truth future latents, how good is the models implicit IDM? 
    - can it relaibly generate the correct actions that caused a specific video sequence?

- add automatic archival saving every 1000 steps
    - in addition to the _latest.pt, save step1000.pt, step2000.pt, etc

- add model size specifications to the paper
    - i think this goes in the experiments section though, since the method should be size agnostic
    - maybe remove the 224x224 and 14x14 numbers from the method then, since the resolution and stuff will be part of the experiments / ablations

- make a front page visualization for the paper
    - Video Latents + Language + State -> DiT -> Future Video Latents + Future Actions
    - Community Datasets
    - Affordable robots and hardware

- critical ablation: does world modelling objective help task success rates?
    - need to train a similar model with no video loss weight
        - need to remove the future latent noise from the model input
    - if the model trains equally well or better, world modelling at this scale doesnt help

- experiment idea: fit chinchilla scaling laws to world modelling objective vs without
    - Compare the 2 scaling curves and see if world modelling helps
    - see if the losses are converging or diverging
    - ask a robot learning person if they think this is a valid experiment


- get evaluation working for SmolVLA, ACT, pi0.5, diffusion policy, basic BC policy (vjepa + mlp)

- add LeRobot citation to paper

- add 3D-RoPE interpolation
    - so the positional ids are the same as training, map the inference resolution onto the training resolution
    - at least i think thats how it works

### Optional / low priority


- add a simple embodiment configuration system
    - or at least add automatic training support for multiple embodiments
    - this could also just be done using lerobot framework afterwards

- add DreamZero-flash noise schedule 

- add context length and action fps variability during training
    - action fps 30 -> 15 -> 10 -> 6 -> 5 randomly during training
    - shift the context length while keeping frame count the same 
    - default context + future length doesnt change, but the number of frames that are context vs denoised can be shifted randomly to improve robustness

- add a model card to lewam-so101-pretrained and other HF models

## Evaluation
- What benchmark to use for testing and comparing to other approaches
    - Libero? 
    - My own suite of comparisons? 
        - eg. also train an ACT, BC, Diffusion policy, smolVLA, etc. (these can all be implemented and trained using lerobot very easily!)
    - for baselines, to be fair, make a BC policy that uses vjepa2 as its visual encoder
- rollout policy on So101 in MuJoCo or Isacc sim 
- Test on real hardware depending on simulated results

## Research
- A key weakness of VJEPA2-AC is that it requires search at inference time to generate actions 
    - if we instead replace the AC (head?) with a diffusion model, we can predict actions and latents directly, making it a WAM

- write up basic motivation in main.tex
- you can look at the TeX source of paper on arxiv for reference on how to make figures for example
- found it pretty important to not update the representation being predicted while its being predicted
    - the learning problem became significantly harder when the embeddings the model needed to predict were constantly shifting

## Notes:
- first training run 
    - 16.8s per step, 18k steps, 64 batch
    - rtx 6000s on vast.ai
    - pretraining on communoty dataset
    - 1.0 action loss weight
    - no competent behaviour in rollout

- second training run:
    - 5.5 sec per step, 10k steps, ~15x4=60 gpu hours (4gpus), 64 batch
    - 4x5090 on vast.ai
    - fine tuning on multi task dataset on my So101
    - 2.0 action weight 
    - overfitting to tasks
    - more robot like motions, but no task success

- third training run:
    - 12 sec per step 5k steps 16x4=64 gpu hours, 240 batch
    - 4x5080 on vast.ai 
    - fine tuning on multi task dataset, with 60 more episodes than trainnig run 2
    - 2.0 action weight
    - 0.5% success rate, managed to pick up a cube twice over the course of an hour


- /dev/tty.usbmodem5B141136531 follower
- /dev/tty.usbmodem5B141125311 leader


