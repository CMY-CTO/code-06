import os
import numpy as np
import torch
import smplx
import sys
sys.path.append("D:\\MingYuan\\A2M\\code")
from render import rendering_utils
from argparse import Namespace
from moviepy.editor import VideoFileClip, clips_array, ColorClip
import glob
import pickle

def get_time_range_from_segment(segment, fps=30, segment_duration=2):
    start_frame = segment * segment_duration * fps
    end_frame = (segment + 1) * segment_duration * fps
    start_time = start_frame / fps
    end_time = end_frame / fps
    return start_time, end_time



def find_intervals_in_range(intervals, start_time, end_time):
    return [interval for interval in intervals if interval['xmax'] > start_time and interval['xmin'] < end_time]

def get_motion_semantic_label(intervals, start_time, end_time):
    extended_start_time = max(0, start_time - 2)
    extended_end_time = end_time + 2
    current_intervals = find_intervals_in_range(intervals, extended_start_time, extended_end_time)
    if not current_intervals:
        return ""
    return " ".join([interval['text'] for interval in current_intervals]).strip()

def get_semantic_score(txt_data, start_time, end_time):
    scores = []
    for entry in txt_data:
        entry_start_time = float(entry[1])
        entry_end_time = float(entry[2])
        if entry_start_time < end_time and entry_end_time > start_time:
            scores.append(float(entry[4]))  # Assuming the semantic score is the fifth element
    if scores:
        return sum(scores) / len(scores)  # 取平均值
    return None

def get_keyword(txt_data, start_time, end_time):
    keywords = []
    for entry in txt_data:
        entry_start_time = float(entry[1])
        entry_end_time = float(entry[2])
        if entry_start_time < end_time and entry_end_time > start_time:
            keyword = entry[5] if len(entry) > 5 else "none"
            keywords.append(keyword)
    if keywords:
        return keywords[0]  # 取第一个出现的 keyword
    return "none"

def arrange_clips(clips, args):
    num_clips = len(clips)
    clips_per_row = 5  # 每行最多显示的clip数量

    # 计算需要的行数
    num_rows = (num_clips + clips_per_row - 1) // clips_per_row

    # 创建一个空白视频剪辑，用于填充不足部分
    blank_clip = ColorClip(size=(args.render_video_width, args.render_video_height), color=(0, 0, 0), duration=1.0 / args.render_video_fps)

    # 创建一个分界线视频剪辑，用于在视频之间添加分界线
    separator_clip = ColorClip(size=(10, args.render_video_height), color=(255, 255, 255), duration=1.0 / args.render_video_fps)

    clip_array = []
    for row in range(num_rows):
        start_index = row * clips_per_row
        end_index = min(start_index + clips_per_row, num_clips)
        row_clips = clips[start_index:end_index]

        # 如果本行的剪辑数少于 clips_per_row，填充空白剪辑
        row_clips += [blank_clip] * (clips_per_row - len(row_clips))

        # 在每个视频之间添加分界线
        row_with_separators = []
        for clip in row_clips:
            if row_with_separators:
                row_with_separators.append(separator_clip)
            row_with_separators.append(clip)

        clip_array.append(row_with_separators)

    return clips_array(clip_array)



# def arrange_clips(clips, args):
#     num_clips = len(clips)

#     # Ensure clip_array is a 2D array
#     if num_clips <= 5:
#         # Single rowf
#         clip_array = [clips]
#     elif num_clips <= 10:
#         # Two rows
#         clip_array = [clips[:5], clips[5:]]
#     else:
#         # Three rows
#         clip_array = [clips[:5], clips[5:10], clips[10:15]]

#     return clips_array(clip_array)

#直接接受从 test_loader 中提取的样本

def visualize_motion_samples(labels, preds, dataloader, args):
    print(f"labels_length{len(labels)},preds_length{len(preds)}")
    print("dataloader_length",len(dataloader))
    for i, sample in enumerate(dataloader):
        # 直接使用批次的数据进行可视化
        visualize_motion(sample, args, labels, f"True_batch_{i}")
        visualize_motion(sample, args, preds, f"Pred_batch_{i}")

def visualize_motion(sample, args, label, label_type):
    with open(f"{args.data_path}\\weights\\vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    pose = sample['pose'][0]
    beta = sample['beta'][0]
    # word = sample['word'][0]

    # 将索引转换为单词
    word_indices = sample['word'][0].tolist()  # 转为列表
    word = []
    for idx in word_indices:
        if 0 <= idx < len(vocab.index2word):
            word.append(vocab.index2word[idx])
        else:
            word.append("<UNK>")  # 未知词处理


    sem = sample['sem'][0]
    trans = sample['trans'][0]
    print("pose",pose.shape)
    print("word",len(word))
    print("sem",sem.shape)
    print("beta",beta.shape)
    print("trans",trans.shape)

    print(f"label_length{len(label)},label_type{label_type}")


    # 确保保存路径存在
    results_save_path = args.output_path
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # 初始化模型
    model = smplx.create(args.smplx_model_path, model_type='smplx',
                         gender='NEUTRAL_2020', use_face_contour=False,
                         num_betas=300, num_expression_coeffs=100,
                         ext='npz', use_pca=False).cpu()

    faces = np.load(f"{args.smplx_model_path}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

    n = pose.shape[0]
    expression = torch.zeros((n, 100)).cpu()

    # expression = sample['expressions'][:n].cpu()
    jaw_pose = pose[:n, 66:69].cpu()
    pose_tensor = pose[:n].cpu()
    transl = trans[:n].cpu()

    output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose,
                   global_orient=pose_tensor[:, :3], body_pose=pose_tensor[:, 3:21*3+3],
                   left_hand_pose=pose_tensor[:, 25*3:40*3], right_hand_pose=pose_tensor[:, 40*3:55*3],
                   leye_pose=pose_tensor[:, 69:72], reye_pose=pose_tensor[:, 72:75], return_verts=True)

    vertices_all = output["vertices"].cpu().detach().numpy()
    vertices_all[:, :, 1] = -vertices_all[:, :, 1]

    seconds = vertices_all.shape[0] // 30 if not args.debug else 1

    folder_output_dir = os.path.join(results_save_path, f"{label_type}_video")
    output_filename = os.path.join(folder_output_dir, f"{label_type}_video.mp4")

    if not os.path.exists(folder_output_dir):
        os.makedirs(folder_output_dir)

    video_file_path = rendering_utils.generate_silent_videos(
        args.render_video_fps,
        args.render_video_width,
        args.render_video_height,
        args.render_concurrent_num,
        'png',
        int(seconds * args.render_video_fps),
        vertices_all,
        [None] * len(vertices_all),
        faces,
        folder_output_dir,
        output_filename,
        sem,
        word,
        labels=label
    )

    if video_file_path is None:
        print(f"Silent video generation failed for {label_type}.")
        return

    clip = VideoFileClip(video_file_path).resize((args.render_video_width, args.render_video_height))
    clip.write_videofile(output_filename, fps=args.render_video_fps)

    print(f"Final {label_type} video saved at {output_filename}")

if __name__ == "__main__":
    npz_file_folders = [
        'D:\\MingYuan\\A2M\\search_grid\\t_4_k_1_upper_position_center',
        'D:\\MingYuan\\A2M\\search_grid\\t_4_k_1_upper_position_1x',
        'D:\\MingYuan\\A2M\\search_grid\\t_4_k_1_upper_position_2x',
        'D:\\MingYuan\\A2M\\search_grid\\t_4_k_1_upper_position_3x'
    ]

    results_save_path = 'D:\\MingYuan\\A2M\\search_grid\\visual_new\\t_4_k_1_upper_position'

    model_folder = 'D:\\MingYuan\\smplx_v1.1\\models'
    txt_folder = 'D:\\MingYuan\\A2M\\2_scott_sem_score'
    textgrid_folder = 'D:\\MingYuan\\A2M\\2_scott_textgrid'

    args = Namespace(
        render_video_fps=30,
        render_video_width=640,
        render_video_height=480,
        render_concurrent_num=4,
        debug=False
    )

    visualize_motion(npz_file_folders, txt_folder, textgrid_folder, results_save_path, model_folder, args)