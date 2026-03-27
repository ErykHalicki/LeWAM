
## Implementation

- create training loop
- make the ground truth actions and patch embeddings are correctly synced
    - VJEPA2 pairs frames, so there are in_frames/2 output frames. Make sure that the relative actions account for this correctly
- test final model parameter count, test feasibility of using 300M VJEPA2.1 checkpoint (ViT L instead of B)
- find a well curated pre training dataset with language captions of videos if possible
- create a decoder for visualizing predicted embeddings
    - can be trained on the embedding - image pairs from just the encoder, and then used to visualize predictor outputs
- test trivial pretraining, overfitting on one video
    - use decoder to test if the predicted outputs make sense
    - can also compare PCA projections of prediced vs ground truth embeddings
- run a slghtly larger pretraining experiment with a real dataset
    - visualize predicted outputs
- come up with training schedule
    - pre training, mid training?, post training
    - maybe the distribution of `tau` can slowly shift higher as training goes on? i.e warming up on low noise amounts, and then progressively increasing the noise mean
        - although there isnt really a strong reason to do this
- pre-compute and cache language embeddings offline (T5GemmaEncoderModel, frozen), no need to run encoder at training time, just load tensors

## Evaluation
- What benchmark to use for testing and comparing to other approaches
    - Libero? 
    - My own suite of comparisons? 
        - eg. also train an ACT, BC, Diffusion policy, smolVLA, etc.
- rollout policy on So101 in MuJoCo or Isacc sim 
- Test on real hardware depending on simulated results

## Research
- update architecture.tex to match real code
    - make architecture diagram for DiT specifically as well
- write up basic motivation in main.tex
