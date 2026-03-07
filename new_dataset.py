import os
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.filterwarnings('ignore', category=PDBConstructionWarning)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

from featurisers import inference_time_create_features


class YjyStyleDataset(Dataset):
    """Dataset for protein domain prediction with 8-channel GPD features.

    Key difference from yylfinal: uses inference_time_create_features(ss_mod=True)
    to generate full 8-channel features instead of manually stacking 3 channels.

    Features output: (8, N, N) per sample
    Labels output: (N,) domain ID tensor (long), 0=unassigned, 1..K=domain ID
    """

    def __init__(self, data_dir, label_file, features_dir=None, max_length=-1,
                 is_single_chain=True, save_features=True, force_reprocess=False,
                 stride_path=None, is_test_set=False, skip_proteins=None,
                 processing_timeout=300):
        """
        Args:
            max_length: Maximum protein length.
                -1 = no limit (default, for batch_size=1; OOM handled in train.py)
                >0 = hard truncate to this length (for batch_size>1)
        """
        self.data_dir = data_dir
        self.label_file = label_file
        self.features_dir = features_dir
        self.max_length = max_length
        self.is_single_chain = is_single_chain
        self.save_features = save_features
        self.force_reprocess = force_reprocess
        self.stride_path = stride_path
        self.is_test_set = is_test_set
        self.skip_proteins = skip_proteins or []
        self.processing_timeout = processing_timeout

        self.parser = PDBParser(QUIET=True)

        if self.save_features and self.features_dir:
            os.makedirs(self.features_dir, exist_ok=True)

        self.pdb_files = self._get_pdb_files()
        LOG.info(f"Found {len(self.pdb_files)} PDB files")

        if self.skip_proteins:
            original_count = len(self.pdb_files)
            self.pdb_files = [
                f for f in self.pdb_files
                if self._get_pdb_id(f) not in self.skip_proteins
            ]
            LOG.info(f"Skipped {original_count - len(self.pdb_files)} proteins")

        self.labels_info = self._load_labels_from_file()
        LOG.info(f"Loaded {len(self.labels_info)} label records")

        if not self.is_test_set:
            self.pdb_files = [
                f for f in self.pdb_files
                if self._get_pdb_id(f) in self.labels_info
            ]
            LOG.info(f"After filtering: {len(self.pdb_files)} labeled PDB files")

    def _get_pdb_files(self):
        pdb_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                if f.lower().endswith(('.pdb', '.cif')):
                    pdb_files.append(os.path.join(root, f))
        return sorted(pdb_files)

    def _get_pdb_id(self, pdb_file):
        return os.path.splitext(os.path.basename(pdb_file))[0].lower()

    def _load_labels_from_file(self):
        """Load domain labels from file.

        Format: PDB_ID NUM_DOMAINS DOMAIN_RANGES
        Example: 1abc 2 100-129,168-292;1-99,130-167;
        """
        labels_info = {}
        if not os.path.exists(self.label_file):
            LOG.warning(f"Label file not found: {self.label_file}")
            return labels_info

        try:
            with open(self.label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        pdb_id = parts[0].lower()
                        domain_ranges_str = ' '.join(parts[2:])

                        domains = []
                        for part in domain_ranges_str.split(';'):
                            part = part.strip()
                            if not part:
                                continue
                            domain_ranges = []
                            for r in part.split(','):
                                r = r.strip()
                                if '-' in r:
                                    try:
                                        start, end = map(int, r.split('-'))
                                        if start >= 0 and end >= start:
                                            domain_ranges.append((start, end))
                                    except ValueError:
                                        continue
                            if domain_ranges:
                                domains.append(domain_ranges)

                        if domains:
                            labels_info[pdb_id] = {
                                'domains': domains,
                                'num_domains': len(domains),
                            }
                    except Exception:
                        continue
        except Exception as e:
            LOG.error(f"Error loading label file: {e}")
            return {}

        LOG.info(f"Successfully loaded {len(labels_info)} label records")
        return labels_info

    def _get_chain_id(self, pdb_file):
        try:
            structure = self.parser.get_structure(
                self._get_pdb_id(pdb_file), pdb_file)
            chains = list(structure.get_chains())
            if not chains:
                return "A"
            return chains[0].get_id()
        except Exception:
            return "A"

    def _generate_features(self, pdb_file, pdb_id, chain_id):
        """Generate 8-channel features using inference_time_create_features.

        KEY FIX: yylfinal's new_dataset.py manually stacked only 3 channels
        (dist, helix, strand). We now call inference_time_create_features(ss_mod=True)
        which produces all 8 channels.
        """
        # Check cache
        if self.save_features and self.features_dir and not self.force_reprocess:
            feat_path = os.path.join(
                self.features_dir, f"{pdb_id}_features.pt")
            if os.path.exists(feat_path):
                try:
                    features = torch.load(feat_path, weights_only=False)
                    # Validate: must have 8 channels
                    if features.dim() == 4 and features.shape[1] == 8:
                        return features
                    else:
                        LOG.info(f"Cached features for {pdb_id} have "
                                 f"{features.shape[1]} channels, need 8. "
                                 f"Regenerating.")
                except Exception as e:
                    LOG.warning(f"Failed to load cached features {pdb_id}: {e}")

        LOG.info(f"Generating 8-channel features for {pdb_id}")
        try:
            # KEY FIX: use inference_time_create_features with ss_mod=True
            features = inference_time_create_features(
                pdb_file, chain=chain_id,
                secondary_structure=True,
                ss_mod=True,
                add_recycling=False,
                add_mask=False,
            )
            # features shape: (1, 8, N, N)

            if self.save_features and self.features_dir:
                feat_path = os.path.join(
                    self.features_dir, f"{pdb_id}_features.pt")
                torch.save(features, feat_path)

            return features

        except Exception as e:
            LOG.error(f"Feature generation failed for {pdb_file}: {e}")
            dummy = torch.zeros(1, 8, 10, 10)
            return dummy

    def _generate_label(self, pdb_id, protein_length):
        """Generate domain label tensor.

        Returns: (protein_length,) long tensor, 0=unassigned, 1..K=domain ID
        """
        # Check cache
        if self.save_features and self.features_dir and not self.force_reprocess:
            label_path = os.path.join(
                self.features_dir, f"{pdb_id}_label.pt")
            if os.path.exists(label_path):
                try:
                    label = torch.load(label_path, weights_only=False)
                    if len(label) == protein_length:
                        return label
                except Exception:
                    pass

        if pdb_id not in self.labels_info:
            if self.is_test_set:
                return None
            return torch.zeros(protein_length, dtype=torch.long)

        info = self.labels_info[pdb_id]
        domains = info['domains']
        label = torch.zeros(protein_length, dtype=torch.long)

        valid_domains = []
        for domain_idx, domain_ranges in enumerate(domains):
            corrected = []
            valid = True
            for start, end in domain_ranges:
                s0 = start - 1  # 1-based to 0-based
                e0 = end - 1
                if s0 < 0 or e0 >= protein_length or s0 > e0:
                    valid = False
                    break
                if e0 - s0 + 1 < 3:
                    valid = False
                    break
                corrected.append((s0, e0))
            if valid and corrected:
                valid_domains.append(corrected)

        for domain_idx, ranges in enumerate(valid_domains):
            domain_label = domain_idx + 1
            for s, e in ranges:
                label[s:e + 1] = domain_label

        # Save label
        if self.save_features and self.features_dir:
            label_path = os.path.join(
                self.features_dir, f"{pdb_id}_label.pt")
            try:
                torch.save(label, label_path)
            except Exception:
                pass

        return label

    def _dense_crop(self, features, label, crop_size):
        """Crop to a contact-dense region with randomness for coverage.

        Picks randomly from top-5 densest windows along the diagonal.
        Each call may return a different region, ensuring the model sees
        different parts of long proteins across epochs.

        Args:
            features: (1, C, N, N) tensor
            label: (N,) tensor
            crop_size: target window size
        Returns:
            (cropped_features, cropped_label, new_length)
        """
        N = features.shape[-1]
        if N <= crop_size:
            return features, label, N

        contact = features[0, 0]  # (N, N) - channel 0 as contact map

        stride = 32
        starts = list(range(0, N - crop_size + 1, stride))
        if starts[-1] != N - crop_size:
            starts.append(N - crop_size)

        scores = []
        for s in starts:
            window = contact[s:s + crop_size, s:s + crop_size]
            scores.append(window.mean().item())

        scores_t = torch.tensor(scores)
        k = min(5, len(scores_t))
        top_indices = scores_t.topk(k).indices
        chosen = top_indices[torch.randint(0, k, (1,)).item()].item()
        start = starts[chosen]
        end = start + crop_size

        return features[:, :, start:end, start:end], label[start:end], crop_size

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_file = self.pdb_files[idx]
        pdb_id = self._get_pdb_id(pdb_file)

        # Try loading cached features + label
        if self.save_features and self.features_dir and not self.force_reprocess:
            feat_path = os.path.join(
                self.features_dir, f"{pdb_id}_features.pt")
            label_path = os.path.join(
                self.features_dir, f"{pdb_id}_label.pt")

            if os.path.exists(feat_path) and os.path.exists(label_path):
                try:
                    features = torch.load(feat_path, weights_only=False)
                    label = torch.load(label_path, weights_only=False)
                    protein_length = features.shape[2]

                    # Dense crop for long proteins (contact-dense + random)
                    if self.max_length > 0 and protein_length > self.max_length:
                        features, label, protein_length = self._dense_crop(
                            features, label, self.max_length)

                    return ({
                        'distance': features.squeeze(0),
                        'protein_length': torch.tensor(
                            protein_length, dtype=torch.long)
                    }, label)
                except Exception:
                    pass

        chain_id = self._get_chain_id(pdb_file)

        # Generate 8-channel features (always full length, no truncation here)
        features = self._generate_features(pdb_file, pdb_id, chain_id)
        protein_length = features.shape[2]

        # Generate label at FULL length first (so it matches complete protein)
        label = self._generate_label(pdb_id, protein_length)
        if label is None:
            label = torch.zeros(protein_length, dtype=torch.long)

        # Align lengths
        if len(label) != protein_length:
            label = torch.zeros(protein_length, dtype=torch.long)

        # Dense crop for long proteins: pick contact-dense region with randomness
        # Cached features are full-length, no re-extraction needed
        if self.max_length > 0 and protein_length > self.max_length:
            features, label, protein_length = self._dense_crop(
                features, label, self.max_length)

        return ({
            'distance': features.squeeze(0),  # (8, N, N)
            'protein_length': torch.tensor(protein_length, dtype=torch.long)
        }, label)

