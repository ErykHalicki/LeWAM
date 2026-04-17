## Implementation

- rollout irl with improved policy
    - task success! very low completion rate, but it does coherent movements
- test new validation set logic

---

- reach out to mees / alexey / jonas / other phd? for feedback

- specify that we use a joint delta action space in the paper

- make a front page visualization for the paper
    - Video Latents + Language + State -> DiT -> Future Video Latents + Future Actions
    - Community Datasets
    - Affordable robots and hardware

- experiment idea: fit chinchilla scaling laws to world modelling objective vs without
    - Compare the 2 scaling curves and see if world modelling helps
    - see if the losses are converging or diverging
    - ask a robot learning person if they think this is a valid experiment

- get evaluation working for SmolVLA, ACT, pi0.5?, diffusion policy, basic BC policy (vjepa + mlp)


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
- v0.1 
    - ~20 gpu hours

- v0.2
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
- ~100 gpu hous

- v0.3
    - ~100 gpu hours

or just count cost by vast.ai credit purchases (currently ~250 USD)

- /dev/tty.usbmodem5B141136531 follower
- /dev/tty.usbmodem5B141125311 leader


