import os
import time
import queue
import threading
import subprocess
import glob
import multiprocessing
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

def distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i]))
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices

def wrap_text(text, width=40):
    print("text",text)
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines), len(lines)

from matplotlib import cm

def render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue, output_dir, semantic_score, motion_semantic_label, labels):
    print(f"fids_length: {len(fids)}, labels_length: {len(labels)}, motion_semantic_label_length: {len(motion_semantic_label)}")
    for fid, (vertices, _) in zip(fids, frame_vertex_pairs):
        fig = plt.figure(figsize=(render_video_width / 100, render_video_height / 100))
        ax = fig.add_subplot(111, projection='3d')

        # 使用 labels 设定颜色 0red1green
        color = '#FF0000' if labels[fid] == 0 else '#00FF00'

        mesh = Poly3DCollection(vertices[faces], alpha=0.1, facecolors=color)
        ax.add_collection3d(mesh)

        ax.set_axis_off()
        ax.view_init(elev=90, azim=90)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1.5, 0.5])
        ax.set_zlim([-2, 0])

        ax.text2D(0.5, -0.05, 'Semantic Score: ', color='#ADD8E6', fontsize=14, ha='center', transform=ax.transAxes)
        ax.text2D(0.5, -0.12, f'{semantic_score[fid]:.2f}', color='#90EE90', fontsize=14, ha='center', transform=ax.transAxes)

        # wrapped_label, wrapped_lines = wrap_text(motion_semantic_label[fid])
        # 确保 motion_semantic_label 是字符串
        label_text = str(motion_semantic_label[fid])
        wrapped_label, wrapped_lines = wrap_text(label_text)
        ax.text2D(0.5, -0.22, 'Semantic Label: ', color='#ADD8E6', fontsize=14, ha='center', transform=ax.transAxes)
        ax.text2D(0.5, -0.22 - 0.07 * wrapped_lines, f'{wrapped_label}', color='#90EE90', fontsize=14, ha='center', transform=ax.transAxes)

        img_path = os.path.join(output_dir, f'frame_{fid}.png')
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        img = Image.open(img_path)
        img = img.resize((render_video_width, render_video_height), Image.ANTIALIAS)
        img.save(img_path)

        fig_queue.put(img_path)

def write_images_from_queue(fig_queue, output_dir, render_tmp_img_filetype):
    while True:
        img_path = fig_queue.get()
        if img_path is None:
            break
        # Process the image (already saved in render_frames_and_enqueue)
        pass

def sub_process_process_frame(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir, semantic_score, motion_semantic_label,labels):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue, output_dir, semantic_score, motion_semantic_label,labels)
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(target=write_images_from_queue, args=(fig_queue, output_dir, render_tmp_img_filetype))
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )

def convert_img_to_mp4(input_pattern, output_file, framerate=30):
    command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file,
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # 确保宽度和高度是2的倍数
        '-y'
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Video conversion successful. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")
        raise

def generate_silent_videos(render_video_fps,
                           render_video_width,
                           render_video_height,
                           render_concurent_nums,
                           render_tmp_img_filetype,
                           frames,
                           vertices_all,
                           vertices1_all,
                           faces,
                           output_dir,
                           output_filename,
                           semantic_score,
                           motion_semantic_label,
                           labels):
    print("silent video label length",len(labels))
    subproc_frame_ids, subproc_vertices = distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame,
            [
                (subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[subprocess_index], subproc_vertices[subprocess_index], faces, output_dir, semantic_score, motion_semantic_label, labels)
                for subprocess_index in range(render_concurent_nums)
            ]
        )

    output_file = os.path.join(output_dir, output_filename)
    try:
        convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"), output_file, render_video_fps)
    except subprocess.CalledProcessError:
        print("Failed to create video. Please check if ffmpeg is installed and supports libx264.")
        return None

    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file