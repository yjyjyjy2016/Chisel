# Chisel: Multi-chain Protein Domain Partitioning

Chisel is a deep learning framework for partitioning complex protein assemblies into functional domains. It uses a hybrid CNN-Attention-GCN architecture with self-conditioned recycling to predict domain boundaries from 3D structure.

## Architecture

```
Input: PDB/CIF structure
         |
    GPD Feature extraction (8 channels)
         |
    + 2 recycling channels (self-conditioning)
         |
  ┌──────────────────────────────────────┐
  │  ChiselBackbone (dilated residual    │
  │  CNN, configurable depth/width)      │
  │  - Dilated residual blocks           │
  │  - Windowed AxialAttention           │
  │    every 4 layers (window=256)       │
  │  - Gradient checkpointing support    │
  └──────────┬───────────────────────────┘
             |
      feat (B,F,N,N)  +  A_hat_raw (B,2,N,N)
             |                    |
     ┌───────┼────────┐          |
     |       |        |          |
   VGAE    pLDDT   Boundary     |
 Refinement  Head    Head       |
     |       |        |          |
  A_refined plddt  boundary    adj + dist
  (B,N,N) (B,1,N,N) (B,1,N)  (B,2,N,N)
```

## Installation

**Requirements:** Python >= 3.8

```bash
# Clone the repository
git clone https://github.com/<your-username>/Chisel.git
cd Chisel

# Install dependencies
pip install -r requirements.txt

# (Optional) Install GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| torch >= 1.12 | Model, training | Yes |
| numpy >= 1.21 | Numerical computing | Yes |
| scipy >= 1.7 | Hungarian matching, distance matrix | Yes |
| biopython >= 1.79 | PDB/CIF parsing | Yes |
| pandas >= 1.3 | Loss logging | Yes |
| tqdm >= 4.62 | Progress bars | Yes |
| mdtraj >= 1.9.7 | Phi/psi, DSSP, centrality (GPD features) | Optional |
| networkx >= 2.6 | Graph centrality | Optional |

> Without `mdtraj` and `networkx`, the model still runs but with degraded feature quality (channels 2-6 zero-filled).

## Quick Start: Inference

```bash
# Fast inference (R=1, default)
python predicttime.py --model_path saved_models/best_model.pt --pdb_path protein.pdb

# Full inference (attention + VGAE + pLDDT + boundary on final round)
python predicttime.py --model_path saved_models/best_model.pt --pdb_path protein.pdb --full

# Batch prediction on a directory
python predicttime.py --model_path saved_models/best_model.pt --pdb_dir /path/to/pdbs/

# Use GPU with fp16
python predicttime.py --model_path saved_models/best_model.pt --pdb_path protein.pdb --device cuda --fp16

# Save results to file
python predicttime.py --model_path saved_models/best_model.pt --pdb_dir /path/to/pdbs/ --output_file results.txt

# Full mode with VGAE-refined adjacency and verbose output
python predicttime.py --model_path saved_models/best_model.pt --pdb_path protein.pdb --full --use_refined --verbose
```

### Inference Modes

Both modes use the same RefinedModel checkpoint:

| Mode | Flag | Speed | What runs |
|------|------|-------|-----------|
| Fast (default) | — | Faster | CNN backbone only (skip attention, no VGAE/pLDDT/boundary) |
| Full | `--full` | Slower | Final round: full model (attention + VGAE + pLDDT + boundary) |

**Output format:**
```
1a59A 2 1-8; 9-377; | pdb idx: 2-9; 10-378
```
`protein_name  num_domains  domain_ranges(1-indexed);  | pdb idx: pdb_residue_ranges`


## Key Features

- **ChiselBackbone**: Dilated residual CNN with configurable depth, width, and axial attention
- **Windowed Axial Attention**: Row + column decomposed attention every 4 CNN layers, computed in windows of 256 to bound memory at O(N*w)
- **VGAE Refinement**: 3-layer GCN encoder (32-dim latent) + bilinear decoder with KL regularization
- **pLDDT Head**: Self-supervised per-residue confidence prediction
- **Boundary Head**: Auxiliary domain boundary detection via Conv1d
- **6-term Loss**: `adj_ce + lr_mse + L_c + vgae + boundary + plddt`
- **Self-conditioned Training**: Recycling mechanism feeds previous predictions back as input
- **Checkpoint Resume**: Full training state recovery (scheduler, epoch, best_val_loss)
- **pIoU Evaluation**: Standard domain segmentation metric

## Project Structure

```
.
├── model.py                    # Model architecture + loss
│                                 - ChiselBackbone (dilated CNN + AxialAttention)
│                                 - RefinedModel (ChiselBackbone + VGAE + pLDDT + Boundary)
│                                 - MyLoss (6-term loss)
│                                 - greedy_domain_assignment()
├── predicttime.py              # Inference (fast / full modes)
├── train.py                    # Training loop with checkpoint resume
├── config.py                   # Argparse configuration
├── new_dataset.py              # Dataset (feature loading)
├── featurisers.py              # 8-channel GPD feature extraction
├── eval_utils.py               # pIoU evaluation
├── constants.py                # Path constants
├── utils/
│   ├── cif2pdb.py              # mmCIF to PDB converter
│   ├── secondary_structure.py  # SS calculation
│   └── pdb_reres.py            # PDB renumbering utility
└── model_modified_v2/          # GPD feature extraction modules
    ├── features/               # Graph + protein feature computation
    └── gpdfeature.py
```


## License

MIT License
