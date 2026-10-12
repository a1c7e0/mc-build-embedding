# Related Work

Papers surfaced during the design discussion (msg.txt) and how our
experiments relate to them.

- **Open-Ended Evolution for Minecraft Building Generation** (2209.03108).
  3D autoencoder latent space + novelty search over Minecraft buildings.
  Closest to our question "does a meaningful latent space exist". Our
  Finding 1-2 reproduce the premise with a much cheaper representation;
  their AE is the natural upgrade path for us.
- **World-GAN: a Generative Model for Minecraft Worlds** (2106.10155).
  block2vec dense block representations + 3D SinGAN-style generator; latent
  edits change world style. Relevant for the "block/structure vector space"
  idea and for the eventual decoder side of our pipeline.
- **NNProc: Approximating Procedural Models of 3D Shapes with Neural
  Networks** (2025). Aligns procedural parameter space and shape space in a
  shared latent. Maps to the "building <-> generator parameters" direction;
  a future step after a working structure autoencoder.
- **Which Structural Constraints Are Learnable? A Regime Map for a Minecraft
  Voxel Generator** (2026). VQ-VAE + autoregressive transformer on 32^3
  voxel grids; asks which structural attributes are controllable. Directly
  relevant to the "constrain outputs to legal blocks/states" design.
- **T2BM / 3D Building Generation in Minecraft via LLMs** (2406.08751).
  Text -> building with an LLM. The contrast case for our architecture
  (LLM as planner only, generator as the workhorse).
- **Interactive Latent Variable Evolution for Minecraft Structures** (FDG
  2022). Latent variable evolution over a generative model of structures;
  another precedent for evolving in latent space.
- **StyleTerrain** — disentangled latent representations for controllable
  procedural terrain generation. Referenced for the "separate style/material/
  scale axes" idea (our Finding 2 shows explicit features already separate
  those factors).

Note: claims above come from the discussion document plus title/abstract
checks; we have not re-derived these results ourselves.
