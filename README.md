# GridLDM: Unified Latent Diffusion for Cross-Domain, Language-Conditioned Power-System Time-Series Synthesis

[//]: # (> A unified generative framework for prompt-conditioned power-system time-series synthesis across multiple domains, including **EV charging**, **commercial load**, **wind generation**, **solar generation**, and **transient voltage response**.)

## Overview

**GridLDM** is a unified latent diffusion framework for controllable power-system time-series generation.  
Instead of training a separate generator for each dataset and each conditioning schema, GridLDM learns a **shared latent generative prior** across heterogeneous power-system domains and conditions generation with **natural-language prompts** through **cross-attention**.

The core idea is:

1. map variable-length time series into a compact common latent space with a **prompt-conditioned autoencoder**,
2. train a **latent diffusion model** in that shared space,
3. use **natural-language prompts** encoded by a pretrained language model to control conditional generation,
4. decode synthetic latents back into the original time-series space.

This repository contains the training and sampling code for the GridLDM pipeline, together with several baseline models used for comparison.

<p align="center">
  <img src="overall_structure.png" alt="GridLDM framework" width="30%"/>
</p>

---

## Method Summary

GridLDM consists of two main stages:

### 1. Prompt-conditioned autoencoder
A Transformer-based conditional autoencoder maps variable-length time series into a fixed latent representation while injecting the text-conditioning signal into both the encoder and decoder.

### 2. Latent diffusion model
A 1D U-Net diffusion model is trained in latent space and uses **cross-attention** over text-token embeddings at each denoising step. During inference, classifier-free guidance is used to improve prompt alignment.

This design makes the model:
- more stable than direct generation in raw signal space,
- easier to reuse across domains with different lengths and temporal structures,
- more flexible than schema-fixed label conditioning.

---

## How to use GridLDM

The recommended workflow for reproducing the project is:

### Step 0. Create environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1. Process the prompts and build text embeddings
Use `data_process.py` to convert prompt JSONL files into token-level BERT embeddings.

```bash
python data_process.py
```

### Step 2. Train the prompt-conditioned autoencoder
Train the multi-domain conditional autoencoder:

```bash
python encoder_decoder_conditional.py
```

This stage learns a shared latent representation across domains and saves:

```text
cond_autoencoder.pt
```

### Step 3. Train GridLDM (latent diffusion)
Train the latent diffusion model in the frozen autoencoder latent space:

```bash
python cross_diffusion_mix.py
```

This stage loads the pretrained autoencoder, encodes time series into latents, and trains the text-conditioned diffusion backbone. The default output checkpoint is:

```text
gridldm_diffusion.pt
```

### Step 4. Run sampling / generation
Generate synthetic time series from prompts with the unified sampling script:

```bash
python gridldm_sampling.py
```

You can also sample only selected domains:

```bash
python gridldm_sampling.py --domains wind solar
python gridldm_sampling.py --domains ev
python gridldm_sampling.py --domains commercial_load transient_voltage
```

Generated samples are saved under domain-specific output folders such as:

```text
samples_wind/
samples_solar/
samples_ev/
samples_comm/
samples_trans/
```

### Step 5. Fine-tune if needed
If you want to adapt GridLDM to a shifted prompt format or a new conditioning style, use the fine-tuning script:

```bash
python cross_diffusion_mix_finetune.py
```

This script freezes the autoencoder and fine-tunes only the diffusion backbone on a small subset of new prompt–data pairs.

### Step 6. Train other baselines
The repository also includes several baseline training scripts:

- **Conditional VAE**
  ```bash
  python cond_VAE.py
  ```

- **Conditional GAN**
  ```bash
  python cond_GAN.py
  ```

- **Direct diffusion in raw signal space**
  ```bash
  python direct_diffusion.py
  ```

These baselines are useful for ablation studies and comparison against GridLDM.

---

## Data Format

### Dataset

We evaluate GridLDM on five power-system time-series domains: **transient voltage response**, **commercial user load**, **EV charging profiles**, **wind generation**, and **solar generation**. The datasets span different temporal resolutions, operating contexts, and conditioning attributes, which makes them suitable for testing unified cross-domain generation.

- **Transient voltage response**: short-term post-disturbance voltage trajectories collected from benchmark power networks.
- **Commercial user load**: daily commercial energy usage traces at **15-minute resolution**.
- **EV charging profiles**: vehicle-level charging records aggregated into normalized **24-hour charging profiles** under different charging-location allocations.
- **Wind generation**: daily wind generation profiles organized by load zone across major U.S. ISOs from **2018 to 2020**, paired with zone-level weather measurements.
- **Solar generation**: daily solar generation profiles organized by load zone across major U.S. ISOs from **2018 to 2020**, paired with zone-level weather measurements.

For the renewable domains, the original data are at **5-minute resolution** and are downsampled to **hourly resolution** to form 24-hour daily profiles. In total, the full experimental setup contains approximately **130k prompt–data pairs**.

> **Important note:** the **commercial load dataset is not included in this repository** because it is confidential. 

For additional details on dataset construction, conditioning attributes, and experiment setup, please refer to the paper.


### Prompt Conditions

GridLDM supports multiple natural-language prompt styles per domain. prompts are stored as JSONL files under `prompts/<domain>/...` and are converted into token embeddings by `data_process.py`.
The same underlying signal can be paired with different prompt formulations, which helps evaluate robustness to linguistic variation and conditional consistency.

For example:
- EV prompts describe **date**, **week type**, **load type**, and **allocation**
- wind / solar prompts describe **zone**, **date**, and weather / pattern statistics
- transient prompts describe **event type**, **bus**, and **network**
- commercial-load prompts describe **user**, **date**, and **weekday/weekend**

---

## Citation

If you use this codebase in your research, please cite the GridLDM paper.

[//]: # (```bibtex)

[//]: # (@article{gridldm,)

[//]: # (  title={GridLDM: Unified Latent Diffusion for Cross-Domain, Language-Conditioned Power-System Time-Series Synthesis},)

[//]: # (  author={Anonymous Author&#40;s&#41;},)

[//]: # (  journal={arXiv / preprint},)

[//]: # (  year={2026})

[//]: # (})

[//]: # (```)

---

## Contact

If you find this repository useful or have questions, please reach out at cyuting@tamu.edu.

