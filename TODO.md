
## Implementation

- convert the hugging face lerobot community dataset (HuggingFaceVLA/community_dataset_v3) to LeRobotDataset v3 format
    - download dataset to euler scratch storage (all v2.1)
    - convert each individual datasets to v2.1->v3.0 (or 2.0->2.1->3.0 if needed)
    - check the distribution of dataset version systematically before downloading if possible
        - check for `"codebase_version": "v2.1",` and 2.0 and 3.0 inside `HuggingFaceVLA/community_dataset_v3/{author}/{dataset_name}/meta/info.json`
    - [Droid conversion guide](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)
    - v2.1 to 3.0 conversion command: `python src/lerobot/scripts/convert_dataset_v21_to_v30.py --repo-id your_id/existing_dataset`
    - upload to hf hub

- Euler details for conversion:
    - scratch dir:  /cluster/scratch/ehalicki/
    - uses slurm management
    - we will need to make a bash script to distribute the work across multiple workers
        - parrallelize v2.1 -> 3.0 conversion

---

- create a training script for lerobot datasets (should work for any lerobot dataset since they all follow te same format)

- make sure the ground truth actions and patch embeddings are correctly synced
    - VJEPA2 pairs frames, so there are in_frames/2 output frames. Make sure that the relative actions account for this correctly

- figure out what happened to my pc?

### Optional / low priority
- create a decoder for visualizing predicted embeddings
    - can be trained on the embedding - image pairs from just the encoder, and then used to visualize predictor outputs

## Evaluation
- What benchmark to use for testing and comparing to other approaches
    - Libero? 
    - My own suite of comparisons? 
        - eg. also train an ACT, BC, Diffusion policy, smolVLA, etc.
    - for baselines, to be fair, make a BC policy that uses vjepa2 as its visual encoder
- rollout policy on So101 in MuJoCo or Isacc sim 
- Test on real hardware depending on simulated results

## Research
- update architecture.tex to match real code
    - make architecture diagram for DiT specifically as well
- write up basic motivation in main.tex
- you can look at the TeX source of paper on arxiv for reference on how to make figures for example
- read about other world models referenced in dreamzero and cite them
