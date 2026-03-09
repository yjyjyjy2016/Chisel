import os
import sys
import torch
import mdtraj as md
import numpy as np
from model_modified_v2.features.graph import compute_rotation_movment, compute_shortestpath_centerilty
from model_modified_v2.features.protein import compute_phipsi_DSSP, get_seq

# 提取 GPD 特征的函数
def get_gpd_feature(pdb_name, length=631):
    top = md.load(pdb_name).topology
    t = md.load(pdb_name)
    distance, movement, quate = compute_rotation_movment(traj=t, top=top, length=length)
    phipsi, DSSP, mask = compute_phipsi_DSSP(top=top, t=t, length=length)
    seqs = get_seq(top=top, length=length)
    path_length, centerity = compute_shortestpath_centerilty(distances=distance, length=length)
    
    distance = distance.float()
    phipsi = torch.from_numpy(phipsi).float()
    DSSP = torch.from_numpy(DSSP).long()
    centerity = torch.from_numpy(centerity).float()

    distance = distance.unsqueeze(0)
    phipsi = phipsi.unsqueeze(0)
    DSSP = DSSP.unsqueeze(0)
    centerity = centerity.unsqueeze(0)

    return distance, phipsi, DSSP, centerity

def convert_gpd_to_stride_format(distance, phipsi, DSSP, centerity):
    """
    将GPD特征转换为与STRIDE特征相同的格式
    GPD特征:
    - distance: (1, N, N) 距离矩阵
    - phipsi: (1, N, 2) phi/psi角度
    - DSSP: (1, N) DSSP二级结构
    - centerity: (1, N) 中心性
    
    STRIDE特征格式:
    - distance_matrix: (N, N) 距离矩阵
    - helix_matrix: (N, N) 螺旋矩阵
    - strand_matrix: (N, N) 折叠矩阵
    - helix_start_end_matrix: (N, N) 螺旋起始终止矩阵
    - strand_start_end_matrix: (N, N) 折叠起始终止矩阵
    """
    # 移除批次维度
    distance_matrix = distance.squeeze(0)  # (N, N)
    phipsi_vals = phipsi.squeeze(0)        # (N, 2)
    dssp_vals = DSSP.squeeze(0)            # (N,)
    centerity_vals = centerity.squeeze(0)  # (N,)
    
    N = distance_matrix.shape[0]
    
    # 根据DSSP值创建螺旋和折叠矩阵
    # DSSP编码: H=α螺旋, E=β折叠, C=卷曲等
    helix_matrix = torch.zeros(N, N)
    strand_matrix = torch.zeros(N, N)
    
    # 对于螺旋 (H, G, I在DSSP中通常代表螺旋)
    helix_mask = (dssp_vals == ord('H')) | (dssp_vals == ord('G')) | (dssp_vals == ord('I'))
    helix_indices = torch.where(helix_mask)[0]
    helix_matrix[helix_indices.unsqueeze(1), helix_indices.unsqueeze(0)] = 1.0
    
    # 对于折叠 (E, B在DSSP中通常代表折叠)
    strand_mask = (dssp_vals == ord('E')) | (dssp_vals == ord('B'))
    strand_indices = torch.where(strand_mask)[0]
    strand_matrix[strand_indices.unsqueeze(1), strand_indices.unsqueeze(0)] = 1.0
    
    # 创建边界矩阵（简化版本）
    helix_start_end_matrix = torch.zeros(N, N)
    strand_start_end_matrix = torch.zeros(N, N)
    
    # 简化的边界检测 - 这里我们只标识连续段的端点
    if helix_indices.numel() > 0:
        diffs = torch.diff(helix_indices)
        breaks = torch.where(diffs > 1)[0]
        segments = []
        start = 0
        for br in breaks:
            segments.append((helix_indices[start].item(), helix_indices[br].item()))
            start = br + 1
        segments.append((helix_indices[start].item(), helix_indices[-1].item()))
        
        for start_idx, end_idx in segments:
            helix_start_end_matrix[start_idx, :] = 1.0
            helix_start_end_matrix[end_idx, :] = 1.0
            helix_start_end_matrix[:, start_idx] = 1.0
            helix_start_end_matrix[:, end_idx] = 1.0
    
    if strand_indices.numel() > 0:
        diffs = torch.diff(strand_indices)
        breaks = torch.where(diffs > 1)[0]
        segments = []
        start = 0
        for br in breaks:
            segments.append((strand_indices[start].item(), strand_indices[br].item()))
            start = br + 1
        segments.append((strand_indices[start].item(), strand_indices[-1].item()))
        
        for start_idx, end_idx in segments:
            strand_start_end_matrix[start_idx, :] = 1.0
            strand_start_end_matrix[end_idx, :] = 1.0
            strand_start_end_matrix[:, start_idx] = 1.0
            strand_start_end_matrix[:, end_idx] = 1.0
    
    # 转换为numpy数组
    distance_matrix = distance_matrix.cpu().numpy()
    helix_matrix = helix_matrix.cpu().numpy()
    strand_matrix = strand_matrix.cpu().numpy()
    helix_start_end_matrix = helix_start_end_matrix.cpu().numpy()
    strand_start_end_matrix = strand_start_end_matrix.cpu().numpy()
    
    return distance_matrix, helix_matrix, strand_matrix, helix_start_end_matrix, strand_start_end_matrix

# 定义一个函数，处理文件夹中的所有 pdb 文件
def process_pdb_folder(input_folder, output_folder, length=631):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith('.pdb')]
    
    for pdb_file in pdb_files:
        pdb_path = os.path.join(input_folder, pdb_file)
        pdb_name = os.path.splitext(pdb_file)[0]  # 获取文件名，去掉扩展名
        
        # 检查输出文件夹中是否已存在该文件的 GPD 特征
        gpd_feature_path = os.path.join(output_folder, f'{pdb_name}.pth')
        if os.path.exists(gpd_feature_path):
            print(f"{gpd_feature_path} already exists, skipping calculation.")
            continue
        
        # 计算 GPD 特征并保存
        try:
           # print(f"Processing {pdb_file}...")
            distance, phipsi, DSSP, centerity = get_gpd_feature(pdb_path, length)
            torch.save((distance, phipsi, DSSP, centerity), gpd_feature_path)
            print(f"Saved GPD features for {pdb_file} to {gpd_feature_path}")
        except Exception as e:
            print(pdb_file)
            print(f"Error processing {pdb_file}: {e}")

# 输入输出文件夹路径
input_folder = '/root/autodl-tmp/yjy/splitdata/iw/WeakConsensus/WeakConsensus'  # 修改为你自己的 pdb 文件夹路径
output_folder = '/root/autodl-tmp/yjy/splitdata/gpd_feature'  # 修改为保存 GPD 特征的文件夹路径

# 执行处理
# process_pdb_folder(input_folder, output_folder)