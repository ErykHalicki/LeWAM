
## Implementation
- pre-compute and cache language embeddings offline (T5GemmaEncoderModel, frozen) — no need to run encoder at training time, just load tensors
- update the DiT to do flow matching
    - we need to output a vector field `[B, K, W, H, D]` where K is chunk size, WH are width height of the feature map
    - and then you integrate the vector field to solve the ODE
- create loss calculations
- create IDM model code (basic transformer)
- create action decoder and state encoder (MLPs)
- create training loop

## Evaluation
- What benchmark to use for testing and comparing to other apporaches
    - Libero? 
    - My own suite of comparisons? 
        - eg. also train an ACT, BC, Diffusion policy, smolVLA, etc.

## Research
- make default viewer of vimtext run OpenPDF nvim command instead of preview
- update architecture.tex to match real code
- write up basic motivation in main.tex
- add facebook DiT to references
- add dreamzero to references
- add LeWorldModel to references
