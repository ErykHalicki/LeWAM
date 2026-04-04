# LeWAM v0.1 Architecture

summary: VJEPA2 + LLM encoder (frozen) -> DiT (diffusion) -> IDM (MSE regression)

DiT trained on SomethingSomethingv2 and LeRobot community dataset (seperatly) with some success

tried to train an IDM only conditioned on VJEPA2 embeddings, but the architecture grew unweildy and had not much precedence.
DiT training was fairly successful even with a small number of steps, but the IDM struggled to even start making loss go lower than a random baseline

Due to technical implmentation cost, debugging, and lack of precedence, I decided to move onto the LeWAM architecture v0.2 using a more principled approach.
Additionally, the lack of an visual grounding in the text encoder was a suboptimal choice to begin with.
