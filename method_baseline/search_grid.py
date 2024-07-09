import os
import numpy as np
import torch
import smplx
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import pickle

joints_list = {
    "beat_smplx_full": {
        "pelvis": 3, "left_hip": 3, "right_hip": 3, "spine1": 3, "left_knee": 3, "right_knee": 3,
        "spine2": 3, "left_ankle": 3, "right_ankle": 3, "spine3": 3, "left_foot": 3, "right_foot": 3,
        "neck": 3, "left_collar": 3, "right_collar": 3, "head": 3, "left_shoulder": 3, "right_shoulder": 3,
        "left_elbow": 3, "right_elbow": 3, "left_wrist": 3, "right_wrist": 3, "jaw": 3, "left_eye_smplhf": 3,
        "right_eye_smplhf": 3, "left_index1": 3, "left_index2": 3, "left_index3": 3, "left_middle1": 3,
        "left_middle2": 3, "left_middle3": 3, "left_pinky1": 3, "left_pinky2": 3, "left_pinky3": 3,
        "left_ring1": 3, "left_ring2": 3, "left_ring3": 3, "left_thumb1": 3, "left_thumb2": 3, "left_thumb3": 3,
        "right_index1": 3, "right_index2": 3, "right_index3": 3, "right_middle1": 3, "right_middle2": 3,
        "right_middle3": 3, "right_pinky1": 3, "right_pinky2": 3, "right_pinky3": 3, "right_ring1": 3,
        "right_ring2": 3, "right_ring3": 3, "right_thumb1": 3, "right_thumb2": 3, "right_thumb3": 3,
    },
    "beat_smplx_upper": {
        "spine1": 3, "spine2": 3, "spine3": 3, "neck": 3, "left_collar": 3, "right_collar": 3,
        "head": 3, "left_shoulder": 3, "right_shoulder": 3, "left_elbow": 3, "right_elbow": 3,
        "left_wrist": 3, "right_wrist": 3,
    }
}

def filter_joints(data, skeleton_type):
    if skeleton_type == "upper":
        upper_indices = [
            3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        ]
        
        # 确保数据是三维的
        if data.ndim == 2:
            data = data.reshape(data.shape[0], -1, 3)

        # 如果数据是CUDA张量，转换为CPU张量然后转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        filtered_data = np.zeros_like(data)
        filtered_indices = [i for i, joint in enumerate(joints_list["beat_smplx_full"]) if joint in joints_list["beat_smplx_upper"]]
        
        for i in filtered_indices:
            filtered_data[:, i, :] = data[:, i, :]

        return filtered_data.reshape(data.shape[0], -1)
    else:
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return data

def axis_angle_to_position(model_path, data_dict, skeleton_type):
    smplx_model = smplx.create(
        model_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
    ).cuda().eval()

    betas = data_dict["betas"]
    poses = data_dict["poses"]
    # trans = data_dict["trans"]
    # exps = data_dict["expressions"]
    # 将表情系数和平移向量置零
    zero_exps = np.zeros_like(data_dict["expressions"])
    zero_trans = np.zeros_like(data_dict["trans"])

    n = poses.shape[0]
    c = poses.shape[1]

    betas = betas.reshape(1, 300)
    betas = np.tile(betas, (n, 1))
    betas = torch.from_numpy(betas).cuda().float()
    poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
    # 用零初始化的数组替换原有数据
    exps = torch.from_numpy(zero_exps.reshape(n, 100)).cuda().float()
    trans = torch.from_numpy(zero_trans.reshape(n, 3)).cuda().float()
    # exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
    # trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()


    max_length = 128
    s, r = n // max_length, n % max_length

    all_tensor = []
    for i in range(s):
        with torch.no_grad():
            joints = smplx_model(
                betas=betas[i*max_length:(i+1)*max_length],
                transl=trans[i*max_length:(i+1)*max_length],
                expression=exps[i*max_length:(i+1)*max_length],
                jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69],
                global_orient=poses[i*max_length:(i+1)*max_length, :3],
                body_pose=poses[i*max_length:(i+1)*max_length, 3:21*3+3],
                left_hand_pose=poses[i*max_length:(i+1)*max_length, 25*3:40*3],
                right_hand_pose=poses[i*max_length:(i+1)*max_length, 40*3:55*3],
                return_verts=True,
                return_joints=True,
                leye_pose=poses[i*max_length:(i+1)*max_length, 69:72],
                reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
            )['joints'][:, :55, :].reshape(max_length, 55*3)
        all_tensor.append(joints)

    if r != 0:
        with torch.no_grad():
            joints = smplx_model(
                betas=betas[s*max_length:s*max_length+r],
                transl=trans[s*max_length:s*max_length+r],
                expression=exps[s*max_length:s*max_length+r],
                jaw_pose=poses[s*max_length:s*max_length+r, 66:69],
                global_orient=poses[s*max_length:s*max_length+r, :3],
                body_pose=poses[s*max_length:s*max_length+r, 3:21*3+3],
                left_hand_pose=poses[s*max_length:s*max_length+r, 25*3:40*3],
                right_hand_pose=poses[s*max_length:s*max_length+r, 40*3:55*3],
                return_verts=True,
                return_joints=True,
                leye_pose=poses[s*max_length:s*max_length+r, 69:72],
                reye_pose=poses[s*max_length:s*max_length+r, 72:75],
            )['joints'][:, :55, :].reshape(r, 55*3)
        all_tensor.append(joints)

    joints = torch.cat(all_tensor, axis=0)
    joints = joints.reshape(n, 55, 3)

    joints = filter_joints(joints, skeleton_type)

    # 确保返回的结果是numpy数组
    if isinstance(joints, torch.Tensor):
        return joints.cpu().numpy()
    else:
        return joints

def compute_acceleration(positions):
    # 计算位置的二阶导数，即加速度
    accelerations = np.diff(positions, n=2, axis=0)
    # 为了保持与原数据长度一致，向前和向后填充0
    pad_width = [(1, 1)] + [(0, 0)] * (accelerations.ndim - 1)
    accelerations = np.pad(accelerations, pad_width, mode='constant', constant_values=0)
    return accelerations

def convert_keypoints_and_filter(data_dict, model_path, skeleton_type, keypoint_type):
    if keypoint_type == "position":
        positions = axis_angle_to_position(model_path, data_dict, skeleton_type)
        return positions
    elif keypoint_type == "acceleration":
        positions = axis_angle_to_position(model_path, data_dict, skeleton_type)
        accelerations = compute_acceleration(positions)
        return accelerations
    else:
        poses = data_dict["poses"]
        if skeleton_type == "upper":
            poses = filter_joints(poses, skeleton_type)
        return poses

def segment_data(npz_files, t, segment_path, model_path, skeleton_type, keypoint_type):
    for npz_file in npz_files:
        data = np.load(npz_file)
        poses = data['poses']
        expressions = data['expressions']
        trans = data['trans']
        betas = data['betas']
        model = data['model']
        gender = data['gender']
        mocap_frame_rate = data['mocap_frame_rate']

        data_dict = {
            "betas": betas,
            "poses": poses,
            "expressions": expressions,
            "trans": trans
        }
        converted_data = convert_keypoints_and_filter(data_dict, model_path, skeleton_type, keypoint_type)

        n_frames = poses.shape[0]
        segment_length = t * 30

        n_segments = n_frames // segment_length
        for i in range(n_segments):
            segment_poses = converted_data[i*segment_length:(i+1)*segment_length]
            segment_expressions = expressions[i*segment_length:(i+1)*segment_length]
            segment_trans = trans[i*segment_length:(i+1)*segment_length]

            npz_data = {
                'betas': betas,
                'poses': segment_poses,
                'expressions': segment_expressions,
                'trans': segment_trans,
                'model': model,
                'gender': gender,
                'mocap_frame_rate': mocap_frame_rate
            }

            output_filename = f"{os.path.splitext(os.path.basename(npz_file))[0]}_segment_{i}.npz"
            output_path = os.path.join(segment_path, output_filename)

            np.savez(output_path, **npz_data)
def cluster_and_visualize(segments_path, k, output_path, t, skeleton_type, keypoint_type):
    all_poses = []
    file_sequence_map = []
    segment_files = [os.path.join(segments_path, f) for f in os.listdir(segments_path) if f.endswith('.npz')]

    for segment_file in segment_files:
        data = np.load(segment_file)
        poses = data['poses']
        all_poses.append(poses)
        file_sequence_map.append(segment_file)

    all_poses = np.array(all_poses)
    
    # notice: n_samples
    # flattened_poses = all_poses.reshape(len(all_poses), -1)
    
    n_samples = all_poses.shape[0]
    flattened_poses = all_poses.reshape(n_samples, -1)  # 每个样本代表一个独立的运动数据

    kmeans = KMeans(n_clusters=k, random_state=42).fit(flattened_poses)
    # TODO:
    # Kmeans dim [n_samples,n_features]
    # distance fn [T*165] [T*55*3] 之间的distance np.l2 norm

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    print(f"KMeans cluster_centers shape: {kmeans.cluster_centers_.shape}")
    print(f"KMeans labels shape: {kmeans.labels_.shape}")
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(flattened_poses)

    print(f"TSNE result shape: {tsne_result.shape}")

    cluster_centers_2d = np.array([tsne_result[labels == i].mean(axis=0) for i in range(k)])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], s=300, c='red', marker='X')

    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)

    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')

    plt.title(f't={t}, k={k}, skeleton={skeleton_type}, keypoint={keypoint_type}')
    plt.savefig(os.path.join(output_path, f'cluster_visualization_t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}.png'))
    plt.close()

    offsets = np.linalg.norm(flattened_poses - cluster_centers[labels], axis=1)
    average_offset = np.mean(offsets)

    center_indices = [np.argmin(np.linalg.norm(flattened_poses - center, axis=1)) for center in cluster_centers]
    center_motion_sequences = [file_sequence_map[idx] for idx in center_indices]

    exceed_1x = np.sum(offsets > average_offset)
    exceed_2x = np.sum(offsets > 2 * average_offset)
    exceed_3x = np.sum(offsets > 3 * average_offset)

    exceed_1x_sequences = [file_sequence_map[i] for i, offset in enumerate(offsets) if offset > average_offset]
    exceed_2x_sequences = [file_sequence_map[i] for i, offset in enumerate(offsets) if offset > 2 * average_offset]
    exceed_3x_sequences = [file_sequence_map[i] for i, offset in enumerate(offsets) if offset > 3 * average_offset]

    with open(os.path.join(output_path, f'exceed_threshold_t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}.txt'), 'w') as f:
        f.write(f"Average offset: {average_offset}\n")
        f.write(f"Cluster centers' motion sequences: {center_motion_sequences}\n")
        f.write(f"Exceed 1x: {exceed_1x}\n")
        for seq in exceed_1x_sequences:
            f.write(f"{seq}\n")
        f.write(f"\nExceed 2x: {exceed_2x}\n")
        for seq in exceed_2x_sequences:
            f.write(f"{seq}\n")
        f.write(f"\nExceed 3x: {exceed_3x}\n")
        for seq in exceed_3x_sequences:
            f.write(f"{seq}\n")

    return exceed_1x_sequences, exceed_2x_sequences, exceed_3x_sequences

def extract_semantics(txt_path, textgrid_path, segment_files, t):
    semantic_scores = []

    for segment_file in segment_files:
        segment_name = os.path.basename(segment_file)
        segment_index = int(segment_name.split('_')[-1].split('.')[0])

        start_time = segment_index * t
        end_time = (segment_index + 1) * t

        txt_file = os.path.join(txt_path, f"{segment_name.split('_segment')[0]}.txt")
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        semantic_score_total = 0
        keyword_list = []
        duration_total = 0

        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                segment_start = float(parts[1])
                segment_end = float(parts[2])
                duration = float(parts[3])
                score = float(parts[4])
                keyword = parts[5] if len(parts) > 5 else 'none'

                overlap_start = max(start_time, segment_start)
                overlap_end = min(end_time, segment_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    semantic_score_total += score * overlap_duration
                    keyword_list.append(keyword)
                    duration_total += overlap_duration

        if duration_total > 0:
            average_semantic_score = semantic_score_total / duration_total
            keyword = max(set(keyword_list), key=keyword_list.count) if keyword_list else 'none'
        else:
            average_semantic_score = 0
            keyword = 'none'

        semantic_scores.append((average_semantic_score, keyword, segment_name))

    return semantic_scores

def extract_semantics_for_full_data(txt_file):
    if not os.path.exists(txt_file):
        print(f"Semantic score file not found: {txt_file}")
        return []

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    semantic_scores = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            duration = float(parts[3])
            score = float(parts[4])
            semantic_scores.append((score, duration))

    return semantic_scores

def compute_average_semantic_score(semantic_scores, t, k, skeleton_type, keypoint_type, threshold):
    if not semantic_scores:
        return {
            't': t,
            'k': k,
            'skeleton_type': skeleton_type,
            'keypoint_type': keypoint_type,
            'threshold': threshold,
            'Average Semantic Score': 0
        }

    total_weighted_score = sum(score * duration for score, duration in semantic_scores)
    total_duration = sum(duration for _, duration in semantic_scores)

    if total_duration == 0:
        average_semantic_score = 0
    else:
        average_semantic_score = total_weighted_score / total_duration

    return {
        't': t,
        'k': k,
        'skeleton_type': skeleton_type,
        'keypoint_type': keypoint_type,
        'threshold': threshold,
        'Average Semantic Score': average_semantic_score
    }

def main(root, model_path, npz_input_path, txt_path, textgrid_path, cluster_output_path):
    npz_files = [os.path.join(npz_input_path, f) for f in os.listdir(npz_input_path) if f.endswith('.npz')]

    t_values = [2, 4, 6, 8, 10]
    k_values = [1, 2, 3, 4, 5, 6]
    skeleton_types = ['full', 'upper']
    keypoint_types = ['rotation', 'position', 'acceleration']
    thresholds = [1, 2, 3]

    combinations = list(product(t_values, k_values, skeleton_types, keypoint_types))
    results = []

    # 缓存文件路径
    cache_file = os.path.join(root, 'search_grid_cache.pkl')

    # 读取缓存
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    for t, k, skeleton_type, keypoint_type in combinations:
        param_key = (t, k, skeleton_type, keypoint_type)

        exceed_threshold_file = os.path.join(cluster_output_path, f"t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}", f'exceed_threshold_t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}.txt')

        if param_key in cache and os.path.exists(exceed_threshold_file):
            exceed_1x_sequences, exceed_2x_sequences, exceed_3x_sequences = cache[param_key]
        else:
            segment_output_path = os.path.join(root, f"search_grid\\segment_output\\t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}")
            os.makedirs(segment_output_path, exist_ok=True)

            segment_data(npz_files, t, segment_output_path, model_path, skeleton_type, keypoint_type)

            cluster_output_subdir = os.path.join(cluster_output_path, f"t_{t}_k_{k}_skeleton_{skeleton_type}_keypoint_{keypoint_type}")
            os.makedirs(cluster_output_subdir, exist_ok=True)
            exceed_1x_sequences, exceed_2x_sequences, exceed_3x_sequences = cluster_and_visualize(segment_output_path, k, cluster_output_subdir, t, skeleton_type, keypoint_type)

            cache[param_key] = (exceed_1x_sequences, exceed_2x_sequences, exceed_3x_sequences)

        for threshold in thresholds:
            if threshold == 1:
                exceed_sequences = exceed_1x_sequences
            elif threshold == 2:
                exceed_sequences = exceed_2x_sequences
            elif threshold == 3:
                exceed_sequences = exceed_3x_sequences

            exceed_semantic_scores = []
            for seq in exceed_sequences:
                txt_file = os.path.join(txt_path, f"{os.path.splitext(os.path.basename(seq))[0].split('_segment')[0]}.txt")
                if os.path.exists(txt_file):
                    exceed_semantic_scores.extend(extract_semantics_for_full_data(txt_file))
                else:
                    print(f"Semantic score file not found: {txt_file}")

            result = compute_average_semantic_score(exceed_semantic_scores, t, k, skeleton_type, keypoint_type, threshold)
            results.append(result)

        # save to Excel
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Average Semantic Score', ascending=False)
        results_df.to_excel('average_semantic_scores.xlsx', index=False)

    # 计算并保存单一参数的平均语义得分
    def compute_single_param_average(df, param):
        return df.groupby(param)['Average Semantic Score'].mean().reset_index()

    # 转换结果为DataFrame
    final_results_df = pd.DataFrame(results)

    # 计算每个单一参数的平均语义得分
    overall_avg = final_results_df['Average Semantic Score'].mean()
    overall_avg_df = pd.DataFrame({'parameter': ['overall'], 'value': ['all'], 'Average Semantic Score': [overall_avg]})

    t_avg_df = compute_single_param_average(final_results_df, 't')
    t_avg_df['parameter'] = 't'
    t_avg_df.columns = ['value', 'Average Semantic Score', 'parameter']

    k_avg_df = compute_single_param_average(final_results_df, 'k')
    k_avg_df['parameter'] = 'k'
    k_avg_df.columns = ['value', 'Average Semantic Score', 'parameter']

    skeleton_type_avg_df = compute_single_param_average(final_results_df, 'skeleton_type')
    skeleton_type_avg_df['parameter'] = 'skeleton_type'
    skeleton_type_avg_df.columns = ['value', 'Average Semantic Score', 'parameter']

    keypoint_type_avg_df = compute_single_param_average(final_results_df, 'keypoint_type')
    keypoint_type_avg_df['parameter'] = 'keypoint_type'
    keypoint_type_avg_df.columns = ['value', 'Average Semantic Score', 'parameter']

    threshold_avg_df = compute_single_param_average(final_results_df, 'threshold')
    threshold_avg_df['parameter'] = 'threshold'
    threshold_avg_df.columns = ['value', 'Average Semantic Score', 'parameter']

    # 合并所有数据
    result_df = pd.concat([overall_avg_df, t_avg_df, k_avg_df, skeleton_type_avg_df, keypoint_type_avg_df, threshold_avg_df], ignore_index=True)

    # 保存合并后的结果到单个Excel文件
    with pd.ExcelWriter('single_param_average_semantic_scores.xlsx') as writer:
        result_df.to_excel(writer, sheet_name='Single Parameter Averages', index=False)

    # 保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

if __name__ == "__main__":
    root = "D:\\MingYuan\\A2M"
    model_path = "D:\\MingYuan\\smplx_v1.1\\models"
    npz_input_path = os.path.join(root, "2_scott")
    txt_path = os.path.join(root, "2_scott_sem_score")
    textgrid_path = os.path.join(root, "2_scott_textgrid")
    cluster_output_path = os.path.join(root, "search_grid\\cluster_output")
    main(root, model_path, npz_input_path, txt_path, textgrid_path, cluster_output_path)