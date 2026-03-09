import os
import sys
import time
import warnings
import argparse

import numpy as np
import torch

from model import RefinedModel, greedy_domain_assignment

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.setrecursionlimit(3000)


# ==============================================================================
# Model loading
# ==============================================================================

def load_model(model_path, device='cpu', filters=64, num_layers=61,
               fp16=False):
    """Load RefinedModel from a training checkpoint."""
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
# Inference
# ==============================================================================

def default_inference(model, features, device='cpu', R=1, K=4):
    model.eval()
    B, C, N, _ = features.shape

    A_prime = torch.zeros(B, 1, N, N, device=device)
    A_bar = torch.zeros(B, 1, N, N, device=device)

    with torch.no_grad():
        for r in range(R):
            new_data = torch.cat([features, A_prime, A_bar], dim=1)
            feat, A_hat_raw = model.cnn(new_data)
            A_bar = A_hat_raw[:, 0:1]

            n_iter = 3 if r == R - 1 else 1
            _, A_prime = greedy_domain_assignment(
                A_bar.squeeze(1), device=device, K_init=K, N_iter=n_iter)
            A_prime = A_prime.unsqueeze(1)

    return A_hat_raw


def full_inference(model, features, device='cpu', R=3, K=4):
    model.eval()
    B, C, N, _ = features.shape

    A_prime = torch.zeros(B, 1, N, N, device=device)
    A_bar = torch.zeros(B, 1, N, N, device=device)

    with torch.no_grad():
        for r in range(R):
            new_data = torch.cat([features, A_prime, A_bar], dim=1)

            if r == R - 1:
                A_hat_raw, A_refined, _, _, plddt, boundary_pred = model(new_data)
            else:
                feat, A_hat_raw = model.cnn(new_data, skip_attention=True)

            A_bar = A_hat_raw[:, 0:1]

            n_iter = 3 if r == R - 1 else 1
            _, A_prime = greedy_domain_assignment(
                A_bar.squeeze(1), device=device, K_init=K, N_iter=n_iter)
            A_prime = A_prime.unsqueeze(1)

    return A_hat_raw, A_refined, plddt, boundary_pred


# ==============================================================================
# Feature extraction
# ==============================================================================

def get_features(pdb_path):
    """Extract 8-channel GPD features for inference."""
    from featurisers import inference_time_create_features
    from featurisers import get_model_structure

    model_structure = get_model_structure(pdb_path)
    chain_ids = [c.id for c in model_structure.get_chains()]
    multi_chain = len(chain_ids) > 1
    chains = chain_ids if multi_chain else chain_ids[0]

    features = inference_time_create_features(
        pdb_path, chain=chains,
        secondary_structure=True,
        ss_mod=True,
        add_recycling=False,
        add_mask=False,
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


def find_domains(adj_matrix, protein_name, pdb_path=None, multi_chain=False,
                 min_segment_length=5):
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

    domains = [d for d in domains if len(d) > 0]

    if pdb_path:
        all_residue_ids = read_protein_ids(pdb_path, chain=True)
        all_residue_ids = [name.strip() for name in all_residue_ids]

        if min_segment_length > 0:
            for domain in domains:
                domain.sort()
                to_remove = set()
                seg_start = domain[0]
                for i in range(1, len(domain)):
                    if (domain[i] != domain[i - 1] + 1 or
                            all_residue_ids[domain[i]][0] !=
                            all_residue_ids[seg_start][0]):
                        seg_len = domain[i - 1] - seg_start + 1
                        if seg_len < min_segment_length:
                            to_remove.update(range(seg_start, domain[i - 1] + 1))
                        seg_start = domain[i]
                seg_len = domain[-1] - seg_start + 1
                if seg_len < min_segment_length:
                    to_remove.update(range(seg_start, domain[-1] + 1))
                domain[:] = [r for r in domain if r not in to_remove]

            domains = [d for d in domains if len(d) > 0]

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

def predict(pdb_path, model, device='cpu', R=1, K=4,
            full=False, use_refined=False):
    """Predict domain boundaries for a PDB file."""
    t0 = time.time()

    features, multi_chain = get_features(pdb_path)

    t_feat = time.time() - t0
    N = features.shape[-1]
    features = features.to(device)

    if next(model.parameters()).dtype == torch.float16:
        features = features.half()

    t1 = time.time()

    plddt_scores = None
    boundary_scores = None

    if full:
        A_hat_raw, A_refined, plddt, boundary_pred = full_inference(
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
        A_hat_raw = default_inference(
            model, features, device=device, R=R, K=K)
        A_bar = A_hat_raw.squeeze(0)[0].cpu()

    t_infer = time.time() - t1

    t2 = time.time()
    _, A_prime = greedy_domain_assignment(
        A_bar.unsqueeze(0), K_init=K, N_iter=3)
    A_prime = A_prime.squeeze(0).cpu().detach().numpy()
    t_assign = time.time() - t2

    mode_str = "full" if full else "default"
    print(f"  Timing (N={N}, {mode_str}): features={t_feat:.1f}s, "
          f"inference={t_infer:.1f}s, assignment={t_assign:.1f}s")

    return A_prime, multi_chain, plddt_scores, boundary_scores


def predict_and_format(pdb_path, model, device='cpu',
                       R=1, K=4, full=False,
                       use_refined=False, verbose=False):
    """Predict domains and return formatted result string."""
    A_prime, multi_chain, plddt_scores, boundary_scores = predict(
        pdb_path, model, device=device,
        R=R, K=K, full=full, use_refined=use_refined)

    protein_name = os.path.basename(pdb_path).rsplit('.', 1)[0]
    result = find_domains(A_prime, protein_name, pdb_path, multi_chain)

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
        description='Protein domain prediction inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to RefinedModel checkpoint (.pt)')
    parser.add_argument('--pdb_path', type=str, default='',
                        help='Path to a single PDB/CIF file')
    parser.add_argument('--pdb_dir', type=str, default='',
                        help='Directory of PDB/CIF files')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--filters', type=int, default=64,
                        help='CNN filters (must match training)')
    parser.add_argument('--num_layers', type=int, default=61,
                        help='Residual layers (must match training)')
    parser.add_argument('--output_file', type=str, default='',
                        help='Output file (append mode)')
    parser.add_argument('--R', type=int, default=1,
                        help='Recycling rounds (default: 1)')
    parser.add_argument('--K', type=int, default=4,
                        help='Initial domain count')
    parser.add_argument('--full', action='store_true',
                        help='Full model: R=3, final round with '
                             'attention + VGAE + pLDDT + boundary')
    parser.add_argument('--use_refined', action='store_true',
                        help='Use VGAE-refined adjacency (requires --full)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 inference (GPU only)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show pLDDT confidence and boundary count '
                             '(--full only)')
    return parser


def main():
    args = get_predict_args().parse_args()
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    if args.use_refined and not args.full:
        print("Warning: --use_refined requires --full, enabling it")
        args.full = True

    if args.full and args.R == 1:
        args.R = 3

    model = load_model(
        args.model_path, device=device,
        filters=args.filters, num_layers=args.num_layers,
        fp16=args.fp16)

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

    mode = "full" if args.full else "default"
    print(f"Processing {len(pdb_files)} file(s) [{mode}, R={args.R}]...")

    success = 0
    failed = 0
    total_time = 0
    for pdb_path in pdb_files:
        try:
            start = time.time()
            result = predict_and_format(
                pdb_path, model, device=device,
                R=args.R, K=args.K,
                full=args.full,
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
