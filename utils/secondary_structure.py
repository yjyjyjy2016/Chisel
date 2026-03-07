import os
import re
import subprocess
import sys

import numpy as np

from constants import REPO_ROOT
import logging
LOG = logging.getLogger(__name__)


def calculate_ss(pdbfile, chain, stride_path, ssfile='pdb_ss'):
    """Calculate secondary structure using STRIDE.
    
    Args:
        pdbfile (str): Path to PDB file
        chain (str): Chain ID
        stride_path (str): Path to STRIDE executable
        ssfile (str): Output file path
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(pdbfile):
        LOG.warning(f"PDB file not found: {pdbfile}")
        return False
        
    try:
        # Check if STRIDE executable exists and is executable
        if not os.path.exists(stride_path):
            LOG.warning(f"STRIDE executable not found at {stride_path}")
            return False
            
        if not os.access(stride_path, os.X_OK):
            LOG.warning(f"STRIDE file is not executable: {stride_path}")
            return False
            
        # Create a temporary copy of PDB file with cleaned coordinates
        tmp_pdb = f"{pdbfile}_tmp.pdb"
        try:
            clean_pdb_for_stride(pdbfile, tmp_pdb)
            pdb_to_use = tmp_pdb
        except:
            LOG.warning(f"Failed to clean PDB file {pdbfile}, using original")
            pdb_to_use = pdbfile
            
        with open(ssfile, 'w') as ssout_file:
            # Check if the chain exists in the PDB file
            chain_exists = False
            with open(pdb_to_use, 'r') as f:
                for line in f:
                    if line.startswith('ATOM') and len(line) > 21:
                        if line[21] == chain or line[21].strip() == chain.strip():
                            chain_exists = True
                            break
            
            if not chain_exists:
                LOG.warning(f"Chain {chain} not found in PDB file {pdbfile}, trying first available chain")
                # Try to find the first available chain
                first_chain = None
                with open(pdb_to_use, 'r') as f:
                    for line in f:
                        if line.startswith('ATOM') and len(line) > 21:
                            first_chain = line[21]
                            break
                
                if first_chain:
                    LOG.info(f"Using chain {first_chain} instead of {chain}")
                    chain = first_chain
                else:
                    LOG.warning(f"No chains found in PDB file {pdbfile}")
                    return False
            
            args = [stride_path, pdb_to_use, '-r' + chain]
            LOG.info(f"Running STRIDE command: {' '.join(args)}")
            
            try:
                result = subprocess.run(args, 
                                    stdout=ssout_file, 
                                    stderr=subprocess.PIPE,
                                    check=True,
                                    encoding='utf-8',
                                    timeout=30)  # Add timeout
                                    
                if result.returncode != 0:
                    LOG.warning(f"STRIDE returned non-zero exit code: {result.returncode}")
                    if result.stderr:
                        LOG.warning(f"STRIDE stderr: {result.stderr}")
                    return False
                    
                return True
                
            except subprocess.TimeoutExpired:
                LOG.warning(f"STRIDE timed out for {pdbfile}")
                return False
            except subprocess.CalledProcessError as e:
                LOG.warning(f"STRIDE failed on {pdbfile} with error: {str(e)}")
                if e.stderr:
                    LOG.warning(f"STRIDE stderr: {e.stderr}")
                return False
            finally:
                if os.path.exists(tmp_pdb):
                    try:
                        os.remove(tmp_pdb)
                    except:
                        pass
                
    except Exception as e:
        LOG.warning(f"Error running STRIDE on {pdbfile}: {str(e)}")
        return False
        
    return False


def clean_pdb_for_stride(input_pdb, output_pdb):
    """Clean PDB file to make it more compatible with STRIDE.

    - Removes alternative locations
    - Keeps only ATOM records
    - Ensures proper formatting of coordinates
    """
    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                # Remove alternative locations
                if line[16] not in [' ', 'A']:
                    continue

                # Clean up coordinates
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    # Format coordinates with proper spacing
                    new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                    f_out.write(new_line)
                except:
                    continue


import re
import numpy as np

def make_ss_matrix(ss_path, nres):
    """Create secondary structure matrices from STRIDE output.

    Args:
        ss_path (str): Path to STRIDE output file
        nres (int): Number of residues

    Returns:
        tuple: (helix_matrix, strand_matrix)
    """
    try:
        helix = np.zeros([nres, nres], dtype=np.float32)
        strand = np.zeros([nres, nres], dtype=np.float32)

        if not os.path.exists(ss_path) or os.path.getsize(ss_path) == 0:
            LOG.warning(f"Empty or missing SS file: {ss_path}")
            return helix, strand

        with open(ss_path) as f:
            lines = f.readlines()

        # 首先尝试从ASG行读取二级结构
        ss_sequence = ''
        for line in lines:
            if line.startswith('ASG'):
                try:
                    ss = line.split()[6]
                    if ss in ['H', 'G', 'I']:  # All types of helices
                        ss_sequence += 'H'
                    elif ss in ['E', 'B']:  # Strands and beta-bridges
                        ss_sequence += 'E'
                    else:
                        ss_sequence += 'C'
                except (IndexError, ValueError) as e:
                    LOG.warning(f"Error parsing ASG line: {line.strip()}")
                    continue

        # 如果ASG行解析失败，尝试从LOC行读取
        if not ss_sequence:
            LOG.info("No ASG assignments found, trying LOC lines...")
            for line in lines:
                if line.startswith('LOC'):
                    try:
                        start_str = re.sub(r'\D', '', line[22:28].strip())
                        end_str = re.sub(r'\D', '', line[40:46].strip())

                        if not start_str or not end_str:
                            continue

                        start = int(float(start_str)) - 1  # 转换为0-based索引
                        end = int(float(end_str))

                        type = line[5:17].strip()

                        if type in ['AlphaHelix', '310Helix', 'PiHelix']:
                            helix[start:end, start:end] = 1
                        elif type in ['Strand', 'Bridge']:
                            strand[start:end, start:end] = 1

                    except (ValueError, IndexError) as e:
                        LOG.warning(f"Error parsing LOC line: {line.strip()}")
                        continue
        else:
            # 使用ASG行的结果填充矩阵
            for i in range(len(ss_sequence)):
                for j in range(len(ss_sequence)):
                    if ss_sequence[i] == 'H' and ss_sequence[j] == 'H':
                        helix[i,j] = 1
                    elif ss_sequence[i] == 'E' and ss_sequence[j] == 'E':
                        strand[i,j] = 1

        return helix, strand

    except Exception as e:
        LOG.warning(f"Error creating SS matrix from {ss_path}: {str(e)}")
        return np.zeros([nres, nres], dtype=np.float32), np.zeros([nres, nres], dtype=np.float32)



def renum_pdb_file(pdb_path, output_pdb_path):
    # 检查 pdb_reres.py 文件是否存在
    pdb_reres_path = REPO_ROOT / 'utils' / 'pdb_reres.py'
    if not os.path.exists(pdb_reres_path):
        # 如果文件不存在，直接复制原文件
        import shutil
        shutil.copy2(pdb_path, output_pdb_path)
        return

    with open(output_pdb_path, "w") as output_file:
        subprocess.run([sys.executable, str(pdb_reres_path), pdb_path],
                       stdout=output_file,
                       check=True,
                       text=True)


def main(chain_ids, pdb_dir, feature_dir, stride_path, reres_path, savedir, job_index=0):
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, '2d_features'), exist_ok=True)
    for chain_id in chain_ids:
        try:
            pdb_path = os.path.join(pdb_dir, chain_id + '.pdb')
            if os.path.exists(pdb_path):
                features = np.load(os.path.join(feature_dir, chain_id + '.npz'))['arr_0']
                nres = features.shape[-1]
                LOG.info("Processing", pdb_path)
                chain = chain_id[4]
                output_pdb_path = os.path.join(savedir, f"{job_index}.pdb") # this gets overwritten to save memory
                file_nres = renum_pdb_file(pdb_path, reres_path, output_pdb_path)
                if nres != file_nres:
                    with open(os.path.join(savedir, 'error.txt'), 'a') as f:
                        msg = f' residue number mismatch (from features) {nres}, (from pdb file) {file_nres}'
                        f.write(chain_id + msg + '\n')
                ss_filepath = os.path.join(savedir, f'pdb_ss{job_index}.txt') # this gets overwritten to save memory
                calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
                helix, strand = make_ss_matrix(ss_filepath, nres=nres)
                np.savez_compressed(
                    os.path.join(*[savedir, '2d_features', chain_id + '.npz']),
                    np.stack((features, helix, strand), axis=0))
        except Exception as e:
            with open(os.path.join(savedir, 'error.txt'), 'a') as f:
                f.write(chain_id + str(e) + '\n')
