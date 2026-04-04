## Implementation

- start another training run
    - use 175gb storage

- create or find simulation environment for evaluation

- get evaluation working for SmolVLA, ACT, pi0.5, diffusion policy, anything available on LeRobot

- calibrate my SO101 arm, test act and smolvla on it in real life

- take the time to start writing up the paper more
    - includes doing more readings

### Optional / low priority

- create a decoder for visualizing predicted embeddings
    - can be trained on the embedding - image pairs from just the encoder, and then used to visualize predictor outputs

- add a simple embodiment configuration system
    - or at least add automatic training support for multiple embodiments
    - this could also just be done using lerobot framework afterwards

- add DreamZero-flash noise schedule 

- add context length and action fps variability during training
    - action fps 30 -> 15 -> 10 -> 6 -> 5 randomly during training
    - shift the context length while keeping frame count the same 
    - default context + future length doesnt change, but the number of frames that are context vs denoised can be shifted randomly to improve robustness

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
- A key weakness of VJEPA2-AC is that it requires search at inference time to generate actions 
    - if we instead replace the AC (head?) with a diffusion model, we can predict actions and latents directly, making it a WAM
- update architecture.tex to match real code
    - make architecture diagram for DiT specifically as well
- write up basic motivation in main.tex
- you can look at the TeX source of paper on arxiv for reference on how to make figures for example
- found it pretty important to not update the representation being predicted while its being predicted
    - the learning problem became significantly harder when the embeddings the model needed to predict were constantly shifting
