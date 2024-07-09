import os
import numpy as np
import torch
import smplx
import rendering_utils
from argparse import Namespace
from moviepy.editor import VideoFileClip, clips_array, ColorClip
import glob

def get_time_range_from_segment(segment, fps=30, segment_duration=2):
    start_frame = segment * segment_duration * fps
    end_frame = (segment + 1) * segment_duration * fps
    start_time = start_frame / fps
    end_time = end_frame / fps
    return start_time, end_time

def load_txt_file(txt_file_path):
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip().split('\t') for line in lines]

def load_textgrid_file(textgrid_file_path):
    with open(textgrid_file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_textgrid_intervals(textgrid_data, tier_name="words"):
    intervals = []
    in_tier = False
    interval = None
    for line in textgrid_data:
        line = line.strip()
        if 'class = "IntervalTier"' in line:
            in_tier = False
        if f'name = "{tier_name}"' in line:
            in_tier = True
        if in_tier:
            if line.startswith('intervals ['):
                if interval:
                    intervals.append(interval)
                interval = {}
            elif line.startswith('xmin') and interval is not None:
                interval['xmin'] = float(line.split('=')[1].strip())
            elif line.startswith('xmax') and interval is not None:
                interval['xmax'] = float(line.split('=')[1].strip())
            elif line.startswith('text') and interval is not None:
                interval['text'] = line.split('=')[1].strip().strip('"')
    if interval:
        intervals.append(interval)
    return intervals

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

def visualize_motion(npz_file_paths, txt_folder, textgrid_folder, results_save_path, model_folder, args):
    # Ensure the results save path exists
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    all_clips = []
    for npz_folder_path in npz_file_paths:
        npz_files = glob.glob(os.path.join(npz_folder_path, '*.npz'))
        folder_clips = []
        for idx, npz_file_path in enumerate(npz_files):
            # Load the motion data from npz file
            data_np_body = np.load(npz_file_path, allow_pickle=True)
            npz_file_name = os.path.basename(npz_file_path)
            base_name = '_'.join(npz_file_name.split('_')[:-2])
            segment = int(npz_file_name.split('_')[-1].split('.')[0])

            # Determine the corresponding txt and TextGrid file paths
            txt_file_path = os.path.join(txt_folder, base_name + '.txt')
            textgrid_file_path = os.path.join(textgrid_folder, base_name + '.TextGrid')

            # Load txt 和 TextGrid file data
            txt_data = load_txt_file(txt_file_path)
            textgrid_data = load_textgrid_file(textgrid_file_path)
            textgrid_intervals = parse_textgrid_intervals(textgrid_data)

            # Get the time range for the segment
            start_time, end_time = get_time_range_from_segment(segment, fps=args.render_video_fps)

            # Get semantic score, motion semantic label 和 keyword
            semantic_score = get_semantic_score(txt_data, start_time, end_time)
            motion_semantic_label = get_motion_semantic_label(textgrid_intervals, start_time, end_time)
            keyword = get_keyword(txt_data, start_time, end_time)

            # Initialize the model
            model = smplx.create(model_folder, model_type='smplx',
                                 gender='NEUTRAL_2020', use_face_contour=False,
                                 num_betas=300, num_expression_coeffs=100,
                                 ext='npz', use_pca=False).cpu()

            faces = np.load(f"{model_folder}/smplx/SMPLX_NEUTRAL_2020.npz", allow_pickle=True)["f"]

            n = data_np_body["poses"].shape[0]

            beta = torch.from_numpy(data_np_body["betas"]).to(torch.float32).unsqueeze(0).cpu()
            beta = beta.repeat(n, 1)
            expression = torch.from_numpy(data_np_body["expressions"][:n]).to(torch.float32).cpu()
            jaw_pose = torch.from_numpy(data_np_body["poses"][:n, 66:69]).to(torch.float32).cpu()
            pose = torch.from_numpy(data_np_body["poses"][:n]).to(torch.float32).cpu()
            transl = torch.from_numpy(data_np_body["trans"][:n]).to(torch.float32).cpu()

            output = model(betas=beta, transl=transl, expression=expression, jaw_pose=jaw_pose,
                           global_orient=pose[:, :3], body_pose=pose[:, 3:21*3+3],
                           left_hand_pose=pose[:, 25*3:40*3], right_hand_pose=pose[:, 40*3:55*3],
                           leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75], return_verts=True)

            vertices_all = output["vertices"].cpu().detach().numpy()

            # Flip the y-axis to make the model upright
            vertices_all[:, :, 1] = -vertices_all[:, :, 1]

            seconds = vertices_all.shape[0] // 30 if not args.debug else 1

            # # Generate unique output filename for each video
            # output_filename = os.path.join(folder_output_dir, npz_file_name.replace('.npz', '.mp4'))

            # Output directory for the specific folder
            folder_output_dir = os.path.join(results_save_path, os.path.basename(npz_folder_path) + "_video")

            # Generate unique output filename for each video
            output_filename = os.path.join(folder_output_dir, npz_file_name.replace('.npz', '.mp4'))

            if not os.path.exists(folder_output_dir):
                os.makedirs(folder_output_dir)

            # Generate silent video
            silent_video_file_path = rendering_utils.generate_silent_videos(
                args.render_video_fps,
                args.render_video_width,
                args.render_video_height,
                args.render_concurrent_num,
                'png',  # render_tmp_img_filetype
                int(seconds * args.render_video_fps),
                vertices_all,
                [None] * len(vertices_all),  # No ground truth vertices
                faces,
                folder_output_dir,
                output_filename,
                semantic_score,
                motion_semantic_label,
                keyword
            )

            if silent_video_file_path is None:
                print(f"Silent video generation failed for {npz_file_path}.")
                continue

            # Adjust the video clip size
            clip = VideoFileClip(silent_video_file_path).resize((args.render_video_width, args.render_video_height))
            folder_clips.append(clip)

        # Combine all clips in the folder into a single video
        if folder_clips:
            combined_clip = arrange_clips(folder_clips, args)
            final_clip_path = os.path.join(folder_output_dir, "final_combined_video.mp4")
            combined_clip.write_videofile(final_clip_path, fps=args.render_video_fps)
            print(f"Final combined video saved at {final_clip_path}")
        else:
            print(f"No videos were generated for folder {npz_folder_path}.")
        all_clips.extend(folder_clips)

    # Combine all folder videos into a single video if needed
    if all_clips:
        combined_clip = arrange_clips(all_clips, args)
        final_clip_path = os.path.join(results_save_path, "final_combined_video.mp4")
        combined_clip.write_videofile(final_clip_path, fps=args.render_video_fps)
        print(f"Final combined video saved at {final_clip_path}")
    else:
        print("No videos were generated.")

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