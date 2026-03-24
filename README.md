# LeWorldActionModel: JEPA based World Action Model

Experiment adapting [LeWorldModel's](https://arxiv.org/pdf/2603.19312v1) next embedding prediction training objective + SIGReg regularization to train a World Action Model. 
Key difference is that instead of predicting p(zt+1 | zt, at) like in LeWM, I use the IDM objective p(at | zt+1, zt) and the next latent prediction objective: p(zt+1 | zt, zt-1, ...). This allows for the model to be pretrained on non-robotics video data like in DreamZero. 

Architecture not finalized yet. Should I jointly train (at, zt+1 | zt, zt-1, ...) or seperate the training objectives and use teacher forcing? I.e predict future latent, and condition an IDM model on the future and current frame, but then input ground truth next frame sometimes? DreamZero uses joint training,
but training them seperately is more intuitive to me since the training can be more easily seperated. 

Sometimes I can let gradients flow through the encoder, prediction model, and IDM (i.e condition on predicted latent NO teacher forcing)
and then the rest of the time only let gradients flow through the IDM and encoder (WITH teacher forcing), skipping the prediction model
and then in the no actions case (eg. internet video pre training), gradient only flows through encoder and IDM
so there are actually a few potential training objectives that can be used.
(at | z^t+1, zt) NO teacher forcing
(at | zt+1, zt) WITH teacher forcing
(zt+1 | zt, zt-1, ...) Video-only pretrainig

**With action labels:**
```math
L = \underbrace{L_{\text{pred}}(z_{t+1}^{gt}, \hat{z}_{t+1})}_{\text{forward model}} + \underbrace{\lambda_{\text{IDM}} \cdot L_{\text{IDM}}(a_t^{gt}, \hat{a}_t)}_{\text{inverse dynamics}} + \underbrace{\lambda_{\text{SIGReg}} \cdot \text{SIGReg}(Z)}_{\text{anti-collapse}}
```

**Video only:**
```math
L = \underbrace{L_{\text{pred}}(z_{t+1}^{gt}, \hat{z}_{t+1})}_{\text{forward model}} + \underbrace{\lambda_{\text{SIGReg}} \cdot \text{SIGReg}(Z)}_{\text{anti-collapse}}
```


## Potential future out of scope directions / exploration:
- what if we also condition the latent / video prediction model on timestamp? I.e, specify how far into the future it should predict?
