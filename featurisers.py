import logging
import os

import Bio.PDB
import numpy as np
import torch
from scipy.spatial import distance_matrix as scipy_distance_matrix

import constants
from utils.cif2pdb import cif2pdb

LOG = logging.getLogger(__name__)

# GPD feature imports — catch ALL exceptions (not just ImportError)
# because submodule internal imports (networkx etc.) can also fail
HAS_GPD = False
try:
    import mdtraj as md
    from model_modified_v2.features.graph import (
        compute_rotation_movment, compute_shortestpath_centerilty)
    from model_modified_v2.features.protein import compute_phipsi_DSSP, get_seq
    HAS_GPD = True
    LOG.info("GPD features enabled (mdtraj + model_modified_v2 loaded)")
except Exception as e:
    LOG.warning(f"GPD features disabled: {e}")


# ==============================================================================
# BioPython helpers
# ==============================================================================

def get_model_structure(structure_path):
    """Returns the Bio.PDB Model object for a PDB or MMCIF file."""
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
            raise ValueError(f"No models found in structure: {structure_path}")
        model = structure[model_ids[0]]

    return model


def get_distance(structure_model, chain='A'):
    """Compute inverse CA distance matrix. Returns (1, N, N)."""
    _3to1 = Bio.PDB.Polypeptide.protein_letters_3to1
    chain_ids = [chain] if isinstance(chain, str) else chain

    alpha_coords = []
    for cid in chain_ids:
        chain_obj = structure_model[cid]
        chain_coords = [
            residue['CA'].get_coord()
            for residue in chain_obj.get_residues()
            if Bio.PDB.is_aa(residue) and 'CA' in residue
            and residue.get_resname() in _3to1
        ]
        alpha_coords.extend(chain_coords)

    alpha_coords = np.array(alpha_coords)
    x = scipy_distance_matrix(alpha_coords, alpha_coords)
    x[x == 0] = x[x > 0].min()
    x = x ** (-1)
    return np.expand_dims(x, axis=0)


# ==============================================================================
# GPD feature extraction
# ==============================================================================

def get_gpd_feature(pdb_name, length=None):
    """Extract GPD features: distance, phi/psi, DSSP, centrality.

    Auto-detects protein length from backbone atom count.
    Returns 4 tensors, each with batch dim prepended.
    """
    if not HAS_GPD:
        raise RuntimeError("GPD features require mdtraj and model_modified_v2. "
                           "Install: pip install mdtraj networkx")

    t = md.load(pdb_name)
    top = t.topology

    # Auto-detect length from backbone atoms
    n_ca = len(top.select("backbone and name CA"))
    n_n = len(top.select("backbone and name N"))
    n_c = len(top.select("backbone and name C"))
    actual_length = min(n_ca, n_n, n_c)

    if length is None:
        length = actual_length
    else:
        length = min(length, actual_length)

    distance_feat, movement, quate = compute_rotation_movment(
        traj=t, top=top, length=length)
    phipsi, DSSP, mask = compute_phipsi_DSSP(top=top, t=t, length=length)
    seqs = get_seq(top=top, length=length)
    path_length, centerity = compute_shortestpath_centerilty(
        distances=distance_feat, length=length)

    distance_feat = distance_feat.float()
    phipsi = torch.from_numpy(phipsi).float()
    DSSP = torch.from_numpy(DSSP).float()  # numeric encoding: H=1,G=2,I=3,E=4,B=5,...
    centerity = torch.from_numpy(centerity).float()

    # Align to consistent length
    N = min(distance_feat.shape[0], phipsi.shape[0],
            DSSP.shape[0], centerity.shape[0])
    distance_feat = distance_feat[:N, :N]
    phipsi = phipsi[:N]
    DSSP = DSSP[:N]
    centerity = centerity[:N]

    return (distance_feat.unsqueeze(0), phipsi.unsqueeze(0),
            DSSP.unsqueeze(0), centerity.unsqueeze(0))


def convert_gpd_to_stride_format(distance_tensor, phipsi, DSSP, centerity):
    """Convert GPD features to N*N matrix format.

    DSSP encoding from features2/protein.py:
        H=1.0, G=2.0, I=3.0 (helices)
        E=4.0, B=5.0 (strands)
        T=6.0, S=7.0, ' '=0.0 (other)

    Returns: (dist_mat, helix_mat, strand_mat,
              centrality_mat, phi_mat, psi_mat)
    """
    dist_mat = distance_tensor.squeeze(0)   # (N, N)
    phipsi_vals = phipsi.squeeze(0)         # (N, 2)
    dssp_vals = DSSP.squeeze(0)             # (N,)
    centerity_vals = centerity.squeeze(0)   # (N,)

    N = min(dist_mat.shape[0], phipsi_vals.shape[0],
            dssp_vals.shape[0], centerity_vals.shape[0])
    dist_mat = dist_mat[:N, :N]
    phipsi_vals = phipsi_vals[:N]
    dssp_vals = dssp_vals[:N]
    centerity_vals = centerity_vals[:N]

    # Helix: H=1, G=2, I=3
    helix_mask = (dssp_vals == 1) | (dssp_vals == 2) | (dssp_vals == 3)
    helix_indices = torch.where(helix_mask)[0]
    helix_matrix = torch.zeros(N, N)
    if len(helix_indices) > 0:
        helix_matrix[helix_indices.unsqueeze(1),
                     helix_indices.unsqueeze(0)] = 1.0

    # Strand: E=4, B=5
    strand_mask = (dssp_vals == 4) | (dssp_vals == 5)
    strand_indices = torch.where(strand_mask)[0]
    strand_matrix = torch.zeros(N, N)
    if len(strand_indices) > 0:
        strand_matrix[strand_indices.unsqueeze(1),
                      strand_indices.unsqueeze(0)] = 1.0

    # Centrality broadcast sum: (N, N)
    centrality_matrix = centerity_vals.unsqueeze(1) + centerity_vals.unsqueeze(0)

    # Phi/Psi broadcast sum: (N, N)
    phi_vals = phipsi_vals[:, 0]
    psi_vals = phipsi_vals[:, 1]
    phi_matrix = phi_vals.unsqueeze(1) + phi_vals.unsqueeze(0)
    psi_matrix = psi_vals.unsqueeze(1) + psi_vals.unsqueeze(0)

    return (dist_mat.cpu().numpy(), helix_matrix.cpu().numpy(),
            strand_matrix.cpu().numpy(), centrality_matrix.cpu().numpy(),
            phi_matrix.cpu().numpy(), psi_matrix.cpu().numpy())


# ==============================================================================
# Relative position encoding
# ==============================================================================

def get_relative_position_encoding(length, normalize=True):
    """Generate (length, length) relative position encoding matrix."""
    positions = np.arange(length)
    rel_pos = (positions[:, None] - positions[None, :]).astype(np.float32)
    if normalize:
        rel_pos = rel_pos / max(1, length - 1)
    return rel_pos


# ==============================================================================
# Main feature generation
# ==============================================================================

def inference_time_create_features(pdb_path, chain="A",
                                    secondary_structure=True,
                                    renumber_pdbs=True,
                                    add_recycling=False,
                                    add_mask=False,
                                    stride_path=constants.STRIDE_EXE,
                                    ss_mod=True,
                                    add_position_encoding=True,
                                    *, model_structure=None):
    """Generate feature tensor for a protein structure.

    When ss_mod=True, produces 8-channel features:
        [dist, helix, strand, phi, psi, centrality, combined_entity, rel_pos]

    When add_recycling=True, 2 zero channels are appended for self-conditioning.

    Returns: (1, C, N, N) torch.Tensor
    """
    if pdb_path.endswith(".cif"):
        pdb_path = cif2pdb(pdb_path)

    if not model_structure:
        model_structure = get_model_structure(pdb_path)

    dist_matrix_feat = get_distance(model_structure, chain=chain)  # (1, N, N)
    n_res = dist_matrix_feat.shape[-1]

    if not secondary_structure:
        features = dist_matrix_feat
        if add_recycling:
            features = np.concatenate(
                (features, np.zeros([2, n_res, n_res], dtype=np.float32)),
                axis=0)
        if add_mask:
            features = np.concatenate(
                (features, np.zeros((1, n_res, n_res), dtype=np.float32)),
                axis=0)
        return torch.Tensor(features[None])

    # --- Extract GPD features (single call, 6 matrices) ---
    helix_full = np.zeros((n_res, n_res), dtype=np.float32)
    strand_full = np.zeros((n_res, n_res), dtype=np.float32)
    centrality_full = np.zeros((n_res, n_res), dtype=np.float32)
    phi_full = np.zeros((n_res, n_res), dtype=np.float32)
    psi_full = np.zeros((n_res, n_res), dtype=np.float32)

    try:
        gpd_raw = get_gpd_feature(pdb_path)
        (_, helix_full, strand_full,
         centrality_full, phi_full, psi_full) = \
            convert_gpd_to_stride_format(*gpd_raw)

        # Align to n_res (GPD auto-length may differ from BioPython n_res)
        N_gpd = helix_full.shape[0]
        N = min(N_gpd, n_res)
        helix_full = helix_full[:N, :N]
        strand_full = strand_full[:N, :N]
        centrality_full = centrality_full[:N, :N]
        phi_full = phi_full[:N, :N]
        psi_full = psi_full[:N, :N]

        # Pad if GPD length < BioPython n_res
        if N < n_res:
            def _pad(m):
                p = np.zeros((n_res, n_res), dtype=np.float32)
                p[:N, :N] = m
                return p
            helix_full = _pad(helix_full)
            strand_full = _pad(strand_full)
            centrality_full = _pad(centrality_full)
            phi_full = _pad(phi_full)
            psi_full = _pad(psi_full)

    except Exception as gpd_err:
        LOG.warning(f"GPD feature extraction failed ({gpd_err}), "
                    f"using zeros for channels 2-6")

    # --- Chain entity matrix ---
    _3to1 = Bio.PDB.Polypeptide.protein_letters_3to1
    chain_ids_per_res = []
    _chains = [chain] if isinstance(chain, str) else chain
    for _cid in _chains:
        for _res in model_structure[_cid].get_residues():
            if (Bio.PDB.is_aa(_res) and 'CA' in _res
                    and _res.get_resname() in _3to1):
                chain_ids_per_res.append(_cid)
    chain_ids_per_res = chain_ids_per_res[:n_res]
    chain_id_arr = np.array(chain_ids_per_res)
    chain_id_matrix = (
        chain_id_arr[:, None] == chain_id_arr[None, :]).astype(np.float32)

    # Normalized residue index difference
    n_chain = len(chain_ids_per_res)
    res_idx = np.arange(n_chain, dtype=np.float32) / max(1, n_chain - 1)
    residue_index_matrix = np.abs(
        res_idx[:, None] - res_idx[None, :]).astype(np.float32)

    # Pad chain matrices if chain_ids_per_res < n_res
    if n_chain < n_res:
        def _pad_chain(m):
            p = np.zeros((n_res, n_res), dtype=np.float32)
            p[:n_chain, :n_chain] = m
            return p
        chain_id_matrix = _pad_chain(chain_id_matrix)
        residue_index_matrix = _pad_chain(residue_index_matrix)

    combined_entity_matrix = 0.5 * chain_id_matrix + 0.5 * residue_index_matrix

    # --- Relative position encoding ---
    if add_position_encoding:
        rel_pos_matrix = get_relative_position_encoding(n_res)
    else:
        rel_pos_matrix = np.zeros((n_res, n_res), dtype=np.float32)

    # --- Stack channels ---
    if ss_mod:
        # 8 channels
        stacked = np.stack((
            dist_matrix_feat[0],   # ch0: inverse CA distance
            helix_full,            # ch1: helix co-membership
            strand_full,           # ch2: strand co-membership
            phi_full,              # ch3: phi angle broadcast sum
            psi_full,              # ch4: psi angle broadcast sum
            centrality_full,       # ch5: centrality broadcast sum
            combined_entity_matrix,# ch6: chain + residue index
            rel_pos_matrix         # ch7: relative position encoding
        ), axis=0)
    else:
        # 6 channels fallback
        stacked = np.stack((
            dist_matrix_feat[0], helix_full, strand_full,
            combined_entity_matrix, residue_index_matrix, rel_pos_matrix
        ), axis=0)

    if add_recycling:
        stacked = np.concatenate(
            (stacked, np.zeros([2, n_res, n_res], dtype=np.float32)),
            axis=0)
    if add_mask:
        stacked = np.concatenate(
            (stacked, np.zeros((1, n_res, n_res), dtype=np.float32)),
            axis=0)

    return torch.Tensor(stacked[None])  # (1, C, N, N)

