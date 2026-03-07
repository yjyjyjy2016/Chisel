import os
import sys
import time
import warnings
import argparse

import numpy as np
import torch
import Bio.PDB
from Bio.PDB import PDBParser

import constants
from utils.secondary_structure import calculate_ss, make_ss_matrix
from model import (ChiselBackbone, RefinedModel, MyLoss, ModelAndLoss,
                   greedy_domain_assignment)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.setrecursionlimit(3000)


# ==============================================================================
# Model loading
# ==============================================================================

def _detect_backbone_config(state_dict):
    # first_block conv weight: (filters, in_channels, 1, 1)
    fb_key = 'model.first_block.0.weight'
    filters = state_dict[fb_key].shape[0]
    ckpt_in_channels = state_dict[fb_key].shape[1]

    # Count residual layers: model.layers.0, model.layers.1, ...
    layer_indices = set()
    for key in state_dict:
        if key.startswith('model.layers.'):
            idx = int(key.split('.')[2])
            layer_indices.add(idx)
    num_layers = max(layer_indices) + 1 if layer_indices else 32

    return {
        'filters': filters,
        'num_layers': num_layers,
        'ckpt_in_channels': ckpt_in_channels,
    }


def load_backbone_model(model_path, device='cpu'):

    state_dict = torch.load(model_path, map_location=device,
                            weights_only=False)

    # Auto-detect architecture from checkpoint weights
    cfg = _detect_backbone_config(state_dict)
    TARGET_IN_CHANNELS = 10  # 8 features + 2 recycling

    # Expand first conv weights: insert zeros between features and recycling
    # Original: [feat(0-4), recycling(5-6)] = 7 channels
    # Target:   [feat(0-4), zeros(5-7), recycling(8-9)] = 10 channels
    fb_key = 'model.first_block.0.weight'
    ckpt_ch = cfg['ckpt_in_channels']
    if ckpt_ch < TARGET_IN_CHANNELS:
        old_weight = state_dict[fb_key]  # (F, 7, 1, 1)
        feat_weight = old_weight[:, :5, :, :]      # (F, 5, 1, 1)
        recycling_weight = old_weight[:, 5:, :, :] # (F, 2, 1, 1)
        zero_pad = torch.zeros(old_weight.shape[0], 3,
                               *old_weight.shape[2:],
                               dtype=old_weight.dtype, device=old_weight.device)
        # Concatenate: feat + zeros + recycling
        state_dict[fb_key] = torch.cat([feat_weight, zero_pad, recycling_weight], dim=1)

    model = ChiselBackbone(
        filters=cfg['filters'], num_layers=cfg['num_layers'],
        in_channels=TARGET_IN_CHANNELS, include_attention=False).to(device)

    loss_weight = {
        'adj_ce_weight': 0.0,
        'lr_mse_weight': 1.0,
        'Lc_weight': 1.0,
        'vgae_weight': 0.0,
        'boundary_weight': 0.0,
        'plddt_weight': 0.0,
    }
    loss_fn = MyLoss(
        device=device, loss_weight=loss_weight,
        positional_encoding='sinusoidal', log=False).to(device)

    model_and_loss = ModelAndLoss(model, loss_fn)

    model_and_loss.load_state_dict(state_dict)
    model_and_loss.eval()

    return model_and_loss


def load_full_model(model_path, device='cpu', filters=64, num_layers=61,
                    fp16=False):
    model = RefinedModel(
        filters=filters,
        num_layers=num_layers,
        in_channels=10,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    if fp16 and device != 'cpu':
        model = model.half()

    return model


# ==============================================================================
# Self-conditioned inference
# ==============================================================================

def backbone_inference(model_and_loss, features, device='cpu', R=8, K=4):
    model_and_loss.train()
    B, C, N, _ = features.shape

    A_prime = torch.zeros(B, 1, N, N, device=device)
    A_bar = torch.zeros(B, 1, N, N, device=device)

    r = torch.randint(0, R + 1, (1,)).item()

    for _ in range(r):
        with torch.no_grad():
            new_data = torch.cat([features, A_prime, A_bar], dim=1)
            output = model_and_loss.model_forward(new_data)
            # ChiselBackbone returns (feat, y_hat); extract y_hat channel 0
            if isinstance(output, tuple):
                output = output[1]
            A_bar = output[:, 0:1]

        _, A_prime = greedy_domain_assignment(
            A_bar.squeeze(1), device=device, K_init=K, N_iter=3)
        A_prime = A_prime.unsqueeze(1)

    # Final forward (with gradients, matching original behavior)
    new_data = torch.cat([features, A_prime, A_bar], dim=1)
    output = model_and_loss.model_forward(new_data)
    if isinstance(output, tuple):
        output = output[1]

    return output


def full_model_inference(model, features, device='cpu', R=8, K=4):
    model.eval()
    B, C, N, _ = features.shape

    A_prime = torch.zeros(B, 1, N, N, device=device)
    A_bar = torch.zeros(B, 1, N, N, device=device)

    with torch.no_grad():
        for r in range(R):
            new_data = torch.cat([features, A_prime, A_bar], dim=1)

            if r == R - 1:
                # Final round: full model
                A_hat_raw, A_refined, _, _, plddt, boundary_pred = model(new_data)
            else:
                # Recycling rounds: backbone only
                feat, A_hat_raw = model.cnn(new_data)

            A_bar = A_hat_raw[:, 0:1]

            n_iter = 3 if r == R - 1 else 1
            _, A_prime = greedy_domain_assignment(
                A_bar.squeeze(1), device=device, K_init=K, N_iter=n_iter)
            A_prime = A_prime.unsqueeze(1)

    return A_hat_raw, A_refined, plddt, boundary_pred


# ==============================================================================
# Feature extraction
# ==============================================================================

def get_model_structure(structure_path):
    """Parse PDB/CIF file, return Bio.PDB Model object."""
    structure_id = os.path.split(structure_path)[-1].split('.')[0]
    if structure_path.endswith('.pdb'):
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(
            structure_id, structure_path)
    elif structure_path.endswith('.cif'):
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(
            structure_id, structure_path)
    else:
        raise ValueError(f'Unrecognized file extension: {structure_path}')
    try:
        model = structure[0]
    except KeyError:
        model_ids = list(structure.child_dict.keys())
        if not model_ids:
            raise ValueError(f"No models in structure: {structure_path}")
        model = structure[model_ids[0]]
    return model


def make_boundary_matrix(ss):
    """Create boundary matrix marking start/end of SS elements."""
    ss_lines = np.zeros_like(ss)
    diag = np.diag(ss)
    if not diag.any() or max(diag) == 0:
        return ss_lines
    padded_diag = np.zeros(len(diag) + 2)
    padded_diag[1:-1] = diag
    diff_before = diag - padded_diag[:-2]
    diff_after = diag - padded_diag[2:]
    start_res = np.where(diff_before == 1)[0]
    end_res = np.where(diff_after == 1)[0]
    ss_lines[start_res, :] = 1
    ss_lines[:, start_res] = 1
    ss_lines[end_res, :] = 1
    ss_lines[:, end_res] = 1
    return ss_lines


def compute_pairwise_distance(structure, chains):
    all_residues = []
    for chain_id in chains:
        chain_res = [res for res in structure[0][chain_id]
                     if 'CA' in res and res.get_full_id()[3][0] == ' ']
        all_residues.extend(chain_res)

    num_residues = len(all_residues)
    dist = np.zeros((num_residues, num_residues), dtype=float)
    for i, res1 in enumerate(all_residues):
        for j, res2 in enumerate(all_residues[i:], start=i):
            d = res1['CA'] - res2['CA']
            dist[i, j] = dist[j, i] = d

    return dist


def predict_secondary_structure(pdb_path, chains=None,
                                stride_path=constants.STRIDE_EXE):
    """Compute STRIDE secondary structure matrices for all chains."""
    if pdb_path.endswith(".cif"):
        structure = Bio.PDB.MMCIFParser(QUIET=True).get_structure(
            'PDB_structure', pdb_path)
    else:
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(
            'PDB_structure', pdb_path)

    if chains is None:
        chains = [chain.id for chain in structure[0]]

    helix_matrix = None
    strand_matrix = None
    helix_boundaries_matrix = None
    strand_boundaries_matrix = None
    first_chain = True

    for chain_id in chains:
        chain = structure[0][chain_id]
        all_residues = [res for res in chain
                        if 'CA' in res and res.get_full_id()[3][0] == ' ']
        if not all_residues:
            continue

        ss_filepath = f"{pdb_path}_{chain_id}_ss.txt"
        calculate_ss(pdb_path, chain_id, stride_path, ssfile=ss_filepath)
        helix, strand = make_ss_matrix(ss_filepath, nres=len(all_residues))
        helix_boundaries = make_boundary_matrix(helix)
        strand_boundaries = make_boundary_matrix(strand)

        if first_chain:
            helix_matrix = helix
            strand_matrix = strand
            helix_boundaries_matrix = helix_boundaries
            strand_boundaries_matrix = strand_boundaries
            first_chain = False
        else:
            helix_matrix = np.block([
                [helix_matrix,
                 np.zeros((helix_matrix.shape[0], helix.shape[1]))],
                [np.zeros((helix.shape[0], helix_matrix.shape[1])),
                 helix]])
            strand_matrix = np.block([
                [strand_matrix,
                 np.zeros((strand_matrix.shape[0], strand.shape[1]))],
                [np.zeros((strand.shape[0], strand_matrix.shape[1])),
                 strand]])
            helix_boundaries_matrix = np.block([
                [helix_boundaries_matrix,
                 np.zeros((helix_boundaries_matrix.shape[0],
                           helix_boundaries.shape[1]))],
                [np.zeros((helix_boundaries.shape[0],
                           helix_boundaries_matrix.shape[1])),
                 helix_boundaries]])
            strand_boundaries_matrix = np.block([
                [strand_boundaries_matrix,
                 np.zeros((strand_boundaries_matrix.shape[0],
                           strand_boundaries.shape[1]))],
                [np.zeros((strand_boundaries.shape[0],
                           strand_boundaries_matrix.shape[1])),
                 strand_boundaries]])

        if os.path.exists(ss_filepath):
            os.remove(ss_filepath)

    return (helix_matrix, strand_matrix,
            helix_boundaries_matrix, strand_boundaries_matrix)


def get_backbone_features(pdb_path, stride_path=constants.STRIDE_EXE):
    parser = PDBParser(QUIET=True)
    pocket_structure = parser.get_structure('pocket', pdb_path)
    model_structure = get_model_structure(pdb_path)
    all_chain_ids = [c.id for c in model_structure.get_chains()]
    multi_chain = len(all_chain_ids) > 1

    if multi_chain:
        residue_numbers = [
            chain_id + res.get_id()[0].strip() +
            str(res.get_id()[1]).strip() + res.get_id()[2].strip()
            for chain_id in all_chain_ids
            for res in pocket_structure[0][chain_id] if 'CA' in res
        ]
    else:
        residue_numbers = [
            res.get_id()[0].strip() +
            str(res.get_id()[1]).strip() + res.get_id()[2].strip()
            for chain_id in all_chain_ids
            for res in pocket_structure[0][chain_id] if 'CA' in res
        ]

    all_residue_ids = read_protein_ids(pdb_path, chain=multi_chain)
    residue_numbers = [n.strip() for n in residue_numbers if 'H_' not in n]
    all_residue_ids = [n.strip() for n in all_residue_ids]

    for res in residue_numbers[:]:
        if res not in all_residue_ids and 'H_' in res:
            residue_numbers.remove(res)

    # Core structural features: distance + secondary structure
    distance_matrix = compute_pairwise_distance(
        pocket_structure, all_chain_ids)
    helix, strand, helix_se, strand_se = predict_secondary_structure(
        pdb_path=pdb_path, chains=all_chain_ids, stride_path=stride_path)

    # Stack into 5 core channels + 3 zero-padded = 8 total (matching GPD layout)
    dist = np.expand_dims(distance_matrix, axis=0)
    helix_ch = np.expand_dims(helix, axis=0)
    strand_ch = np.expand_dims(strand, axis=0)
    helix_se_ch = np.expand_dims(helix_se, axis=0)
    strand_se_ch = np.expand_dims(strand_se, axis=0)

    N = distance_matrix.shape[0]
    input_data = np.concatenate(
        [dist, helix_ch, strand_ch, helix_se_ch, strand_se_ch,
         np.zeros((3, N, N))], axis=0)  # pad to 8 channels
    features = torch.tensor(
        input_data, dtype=torch.float32).unsqueeze(0)  # (1, 8, N, N)

    return features, multi_chain, distance_matrix


def get_full_features(pdb_path, stride_path=constants.STRIDE_EXE):
    """Extract extended 8-channel GPD features for full model inference."""
    from featurisers import inference_time_create_features
    from featurisers import get_model_structure as feat_get_model_structure

    model_structure = feat_get_model_structure(pdb_path)
    chain_ids = [c.id for c in model_structure.get_chains()]
    multi_chain = len(chain_ids) > 1
    chains = chain_ids if multi_chain else chain_ids[0]

    features = inference_time_create_features(
        pdb_path, chain=chains,
        secondary_structure=True,
        ss_mod=True,
        add_recycling=False,
        add_mask=False,
        stride_path=stride_path,
        model_structure=model_structure,
    )
    return features, multi_chain


# ==============================================================================
# Domain output formatting
# ==============================================================================

def read_protein_ids(filename, chain=None):
    """Read residue IDs from PDB file."""
    if chain is None:
        chain_ids = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    chain_id = line[21]
                    if chain_id not in chain_ids:
                        chain_ids.append(chain_id)
            if len(chain_ids) > 1:
                chain = True
    ids = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                if chain:
                    atom_id = line[21] + line[22:30].strip()
                else:
                    atom_id = line[22:30].strip()
                if atom_id not in ids:
                    ids.append(atom_id)
    return ids


def find_domains(adj_matrix, protein_name, pdb_path=None, multi_chain=False, distance_matrix=None):
    visited = [False] * len(adj_matrix)
    domains = []

    def dfs(i, current_domain):
        visited[i] = True
        current_domain.append(i)
        for j, is_connected in enumerate(adj_matrix[i]):
            if is_connected and not visited[j]:
                dfs(j, current_domain)

    for i in range(len(adj_matrix)):
        if not visited[i]:
            current_domain = []
            dfs(i, current_domain)
            if len(current_domain) > 1:
                domains.append(current_domain)

    # Assign orphan residues to nearest domain
    for i in range(len(adj_matrix)):
        if not visited[i]:
            closest_domain = None
            min_distance = float('inf')
            for domain in domains:
                for idx in domain:
                    # Use spatial distance if available, otherwise sequence distance
                    if distance_matrix is not None:
                        distance = distance_matrix[i, idx]
                    else:
                        distance = abs(idx - i)
                    if distance < min_distance:
                        min_distance = distance
                        closest_domain = domain
            if closest_domain is not None:
                closest_domain.append(i)

    if pdb_path:
        all_residue_ids = read_protein_ids(pdb_path, chain=multi_chain)
        all_residue_ids = [name.strip() for name in all_residue_ids]

        # Post-process: merge or remove short domains/fragments (≤ 3 residues)
        if multi_chain:
            changed = True
            while changed:
                changed = False
                short_domain_indices = []
                for d_idx, domain in enumerate(domains):
                    if len(domain) <= 3:
                        short_domain_indices.append(d_idx)

                for d_idx in reversed(short_domain_indices):
                    short_domain = domains[d_idx]
                    # Find best target domain to merge into (same chain, closest)
                    best_target = None
                    best_dist = float('inf')
                    for t_idx, target in enumerate(domains):
                        if t_idx == d_idx or len(target) <= 3:
                            continue
                        for si in short_domain:
                            for ti in target:
                                # Same chain check
                                if all_residue_ids[si][0] == all_residue_ids[ti][0]:
                                    d = abs(si - ti)
                                    if d < best_dist:
                                        best_dist = d
                                        best_target = t_idx
                    if best_target is not None:
                        domains[best_target].extend(short_domain)
                        domains.pop(d_idx)
                        changed = True
                    else:
                        # No same-chain target found, remove entirely
                        domains.pop(d_idx)
                        changed = True

            # Also clean up short segments within larger domains
            for domain in domains:
                domain.sort()
                segments = []
                start = domain[0]
                for i in range(1, len(domain)):
                    if (domain[i] != domain[i - 1] + 1 or
                            all_residue_ids[start][0] != all_residue_ids[domain[i]][0]):
                        segments.append((start, domain[i - 1], all_residue_ids[start][0]))
                        start = domain[i]
                segments.append((start, domain[-1], all_residue_ids[start][0]))

                short_segments = [i for i, (s, e, c) in enumerate(segments) if e - s + 1 <= 3]

                for seg_idx in reversed(short_segments):
                    s, e, chain = segments[seg_idx]

                    merged = False
                    # Try merge with next same-chain segment
                    if seg_idx + 1 < len(segments):
                        ns, ne, nc = segments[seg_idx + 1]
                        if chain == nc:
                            segments[seg_idx + 1] = (s, ne, chain)
                            segments.pop(seg_idx)
                            merged = True
                    # Try merge with previous same-chain segment
                    if not merged and seg_idx > 0:
                        ps, pe, pc = segments[seg_idx - 1]
                        if chain == pc:
                            segments[seg_idx - 1] = (ps, e, chain)
                            segments.pop(seg_idx)
                            merged = True
                    # Cannot merge, remove
                    if not merged:
                        residues_to_remove = set(range(s, e + 1))
                        domain[:] = [r for r in domain if r not in residues_to_remove]

        # Remove empty domains
        domains = [d for d in domains if len(d) > 0]

        # Format output
        # Always use chain=True so IDs are chain-prefixed (e.g. "A4")
        # and [0] correctly compares chain letters, not residue number digits
        all_residue_ids = read_protein_ids(pdb_path, chain=True)
        all_residue_ids = [name.strip() for name in all_residue_ids]

        output_strs = []
        for domain in domains:
            domain.sort()
            parts = []
            start = domain[0]
            for i in range(1, len(domain)):
                if (domain[i] != domain[i - 1] + 1 or
                        all_residue_ids[start][0] !=
                        all_residue_ids[domain[i]][0]):
                    parts.append(f'{start + 1}-{domain[i - 1] + 1}')
                    start = domain[i]
            parts.append(f'{start + 1}-{domain[-1] + 1}')
            output_strs.append(','.join(parts))

        domain_output = '; '.join(output_strs)
        result = f'{protein_name} {len(domains)} {domain_output};'

        pdb_domain_strs = []
        for domain in domains:
            parts = []
            start = domain[0]
            for i in range(1, len(domain)):
                if (domain[i] != domain[i - 1] + 1 or
                        all_residue_ids[start][0] !=
                        all_residue_ids[domain[i]][0]):
                    parts.append(
                        f'{all_residue_ids[start]}-'
                        f'{all_residue_ids[domain[i - 1]]}')
                    start = domain[i]
            parts.append(
                f'{all_residue_ids[start]}-{all_residue_ids[domain[-1]]}')
            pdb_domain_strs.append(','.join(parts))
        result += ' | pdb idx: ' + '; '.join(pdb_domain_strs)
    else:
        output_strs = []
        for domain in domains:
            domain.sort()
            parts = []
            start = domain[0]
            for i in range(1, len(domain)):
                if domain[i] != domain[i - 1] + 1:
                    parts.append(f'{start + 1}-{domain[i - 1] + 1}')
                    start = domain[i]
            parts.append(f'{start + 1}-{domain[-1] + 1}')
            output_strs.append(','.join(parts))

        domain_output = '; '.join(output_strs)
        result = f'{protein_name} {len(domains)} {domain_output};'

    return result


# ==============================================================================
# Prediction pipeline
# ==============================================================================

def predict(pdb_path, model, device='cpu',
            stride_path=constants.STRIDE_EXE, R=8, K=4,
            full_model=False, use_refined=False):
    t0 = time.time()

    distance_matrix = None
    if full_model:
        features, multi_chain = get_full_features(
            pdb_path, stride_path=stride_path)
    else:
        features, multi_chain, distance_matrix = get_backbone_features(
            pdb_path, stride_path=stride_path)

    t_feat = time.time() - t0
    N = features.shape[-1]
    features = features.to(device)

    if full_model and next(model.parameters()).dtype == torch.float16:
        features = features.half()

    t1 = time.time()

    plddt_scores = None
    boundary_scores = None

    if full_model:
        A_hat_raw, A_refined, plddt, boundary_pred = full_model_inference(
            model, features, device=device, R=R, K=K)

        if use_refined:
            A_bar = A_refined.squeeze(0).cpu()
        else:
            A_bar = A_hat_raw.squeeze(0)[0].cpu()

        if plddt is not None:
            plddt_scores = torch.diagonal(
                plddt[0, 0], dim1=-2, dim2=-1).cpu().numpy()
        if boundary_pred is not None:
            boundary_scores = boundary_pred[0, 0].cpu().numpy()
    else:
        output = backbone_inference(
            model, features, device=device, R=R, K=K)
        A_bar = output.squeeze(0)[0].cpu()

    t_infer = time.time() - t1

    t2 = time.time()
    _, A_prime = greedy_domain_assignment(
        A_bar.unsqueeze(0), K_init=K, N_iter=3)
    A_prime = A_prime.squeeze(0).cpu().detach().numpy()
    t_assign = time.time() - t2

    mode_str = "full" if full_model else "backbone"
    print(f"  Timing (N={N}, {mode_str}): features={t_feat:.1f}s, "
          f"inference={t_infer:.1f}s, assignment={t_assign:.1f}s")

    return A_prime, multi_chain, plddt_scores, boundary_scores, distance_matrix


def predict_and_format(pdb_path, model, device='cpu',
                       stride_path=constants.STRIDE_EXE,
                       R=8, K=4, full_model=False, use_refined=False,
                       verbose=False):
    """Predict domains and return formatted result string."""
    A_prime, multi_chain, plddt_scores, boundary_scores, distance_matrix = predict(
        pdb_path, model, device=device, stride_path=stride_path,
        R=R, K=K, full_model=full_model, use_refined=use_refined)

    # For multi-chain proteins, break spurious connections at chain boundaries
    # Only break edges between different chains if they are spatially distant
    if multi_chain and pdb_path and distance_matrix is not None:
        all_residue_ids = read_protein_ids(pdb_path, chain=multi_chain)
        all_residue_ids = [name.strip() for name in all_residue_ids]
        # Extract chain ID (first character for multi-chain)
        chain_labels = [res_id[0] for res_id in all_residue_ids]

        # Break cross-chain edges if spatial distance > threshold (e.g., 8Å)
        # This prevents sequence-adjacent but spatially-distant residues from merging
        distance_threshold = 8.0  # Angstroms
        N = A_prime.shape[0]
        for i in range(N):
            for j in range(N):
                if chain_labels[i] != chain_labels[j]:
                    if distance_matrix[i, j] > distance_threshold:
                        A_prime[i, j] = 0

    protein_name = os.path.basename(pdb_path).rsplit('.', 1)[0]
    result = find_domains(A_prime, protein_name, pdb_path, multi_chain,
                         distance_matrix=distance_matrix)

    if verbose and plddt_scores is not None and boundary_scores is not None:
        mean_conf = plddt_scores.mean()
        n_boundaries = (boundary_scores > 0.5).sum()
        result += f'  [conf={mean_conf:.3f}, boundaries={n_boundaries}]'

    return result


# ==============================================================================
# CLI
# ==============================================================================

def get_predict_args():
    parser = argparse.ArgumentParser(
        description='Protein domain prediction inference '
                    '(backbone-only fast mode or full model)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--pdb_path', type=str, default='',
                        help='Path to a single PDB/CIF file')
    parser.add_argument('--pdb_dir', type=str, default='',
                        help='Directory of PDB/CIF files')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--filters', type=int, default=64,
                        help='CNN filters for full model (must match training)')
    parser.add_argument('--num_layers', type=int, default=61,
                        help='Residual layers for full model (must match training)')
    parser.add_argument('--stride_path', type=str,
                        default=constants.STRIDE_EXE,
                        help='Path to STRIDE executable')
    parser.add_argument('--output_file', type=str, default='',
                        help='Output file (append mode)')
    parser.add_argument('--R', type=int, default=8,
                        help='Recycling rounds')
    parser.add_argument('--K', type=int, default=4,
                        help='Initial domain count')
    parser.add_argument('--full_model', action='store_true',
                        help='Use full model (RefinedModel with VGAE + pLDDT '
                             '+ boundary + 8-channel GPD features). '
                             'Default: backbone-only with core features')
    parser.add_argument('--use_refined', action='store_true',
                        help='Use VGAE-refined adjacency (requires --full_model)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 inference (GPU only, full_model only)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show pLDDT confidence and boundary count '
                             '(full_model only)')
    return parser


def main():
    args = get_predict_args().parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    if args.use_refined and not args.full_model:
        print("Warning: --use_refined requires --full_model, enabling it")
        args.full_model = True

    # Load model
    if args.full_model:
        model = load_full_model(
            args.model_path, device=device,
            filters=args.filters, num_layers=args.num_layers,
            fp16=args.fp16)
    else:
        model = load_backbone_model(args.model_path, device=device)

    # Collect PDB files
    pdb_files = []
    if args.pdb_path and os.path.isfile(args.pdb_path):
        pdb_files = [args.pdb_path]
    elif args.pdb_dir and os.path.isdir(args.pdb_dir):
        pdb_files = [
            os.path.join(args.pdb_dir, f)
            for f in sorted(os.listdir(args.pdb_dir))
            if f.lower().endswith(('.pdb', '.cif'))
        ]
    elif args.pdb_path and os.path.isdir(args.pdb_path):
        pdb_files = [
            os.path.join(args.pdb_path, f)
            for f in sorted(os.listdir(args.pdb_path))
            if f.lower().endswith(('.pdb', '.cif'))
        ]

    if not pdb_files:
        print("No PDB files found. Use --pdb_path or --pdb_dir.")
        return

    print(f"Processing {len(pdb_files)} file(s)...")

    success = 0
    failed = 0
    total_time = 0
    for pdb_path in pdb_files:
        try:
            start = time.time()
            result = predict_and_format(
                pdb_path, model, device=device,
                stride_path=args.stride_path,
                R=args.R, K=args.K,
                full_model=args.full_model,
                use_refined=args.use_refined,
                verbose=args.verbose)
            elapsed = time.time() - start
            total_time += elapsed
            print(f"{result}  ({elapsed:.1f}s)")

            if args.output_file:
                with open(args.output_file, 'a') as f:
                    f.write(result + '\n')
            success += 1
        except Exception as e:
            failed += 1
            print(f"Error: {pdb_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {success} succeeded, {failed} failed, "
          f"total={total_time:.1f}s, avg={total_time/max(success,1):.1f}s")


if __name__ == '__main__':
    main()
