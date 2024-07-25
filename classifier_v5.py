import time

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import textgrid as tg
import logging

import lmdb

import pyarrow
import pickle
import joblib
import chardet

from collections import defaultdict
import pandas as pd

import math
from termcolor import colored
import shutil

import smplx
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from loguru import logger

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("D:\\MingYuan\\A2M\\code")

from dataloaders.data_tools import joints_list
from dataloaders.build_vocab import *
from dataloaders.build_vocab import Vocab

from utils import rotation_conversions as rc
from utils import other_tools
# # 设置CUDA_LAUNCH_BLOCKING环境变量
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class MotionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=4):
        super(MotionEncoder, self).__init__()
        channels = [output_dim] * n_layers
        layers = [
            nn.Conv1d(input_dim, channels[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_layers):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)

    def forward(self, x):
        return self.main(x.permute(0, 2, 1)).permute(0, 2, 1)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, data_path, freeze_pretrained=True):
        super(TextEncoder, self).__init__()
        with open(f"{data_path}\\weights\\vocab.pkl", 'rb') as f:
            lang_model = pickle.load(f)
            pre_trained_embedding = lang_model.word_embedding_weights
            pre_trained_embedding_tensor = torch.FloatTensor(pre_trained_embedding)

        self.vocab_size = pre_trained_embedding.shape[0]
        self.pre_embedding = nn.Embedding.from_pretrained(pre_trained_embedding_tensor, freeze=freeze_pretrained)
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # self.seq_len = seq_len

    def forward(self, x):
        # 检查输入索引是否在词汇表范围内
        if torch.max(x) >= self.vocab_size or torch.min(x) < 0:
            print(f"Input tensor out of bounds: x shape: {x.shape}, Max index: {torch.max(x)}, Min index: {torch.min(x)}, Vocab size: {self.vocab_size}")
            raise ValueError(f"Input indices are out of bounds. Max index: {torch.max(x)}, Min index: {torch.min(x)}, Vocab size: {self.vocab_size}")

        # 打印输入张量的最大最小值
        # print(f"Input tensor max: {torch.max(x)}, min: {torch.min(x)}, Vocab size: {self.vocab_size}")

        # 确保索引值在范围内，并转换为 long 类型
        x = torch.clamp(x, min=0, max=self.vocab_size - 1).long()

        # 再次打印调试信息以确认 clamp 后的值
        # print(f"After clamping - Input tensor: {x}")

        # 确保输入张量的数据类型为 long
        # print(f"Text Input tensor dtype: {x.dtype}")

        # 打印输入张量的形状
        # print(f"Text Input tensor shape: {x.shape}")

        # 使用断言确保所有索引值在词汇表范围内
        assert torch.all(x >= 0) and torch.all(x < self.vocab_size), f"Index out of range: {x}"

        # 在调用 embedding 之前同步 CUDA 设备
        torch.cuda.synchronize()

        # 打印 embedding 层的参数
        # print(f"Embedding weight shape: {self.embedding.weight.shape}")
        # print(f"Embedding weight max: {torch.max(self.embedding.weight)}, min: {torch.min(self.embedding.weight)}")

        try:
            x = self.embedding(x).permute(0, 2, 1)
        except Exception as e:
            print(f"Error during embedding: {e}")
            print(f"Input tensor: {x}")
            raise  # 重新引发异常，停止执行

        # 打印 embedding 后的形状
        print("embedding_shape", x.shape)

        x = self.conv(x).permute(0, 2, 1)  # [batch_size, embed_dim, seq_len] -> [batch_size, seq_len, output_dim]
        print("embedding_conv_shape", x.shape)

        return x

class MotionTextClassifier(nn.Module):
    def __init__(self, motion_input_dim, text_input_dim, embed_dim, text_output_dim, hidden_size, num_classes):
        super(MotionTextClassifier, self).__init__()
        self.motion_encoder = MotionEncoder(motion_input_dim, 256, 4)
        self.text_encoder = TextEncoder(text_input_dim, embed_dim, text_output_dim, data_path="D:\\MingYuan\\Dataset\\beat_v2.0.0\\beat_english_v2.0.0", freeze_pretrained=True)

        self.fc = nn.Sequential(
            nn.Linear(256 + 256, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, motion, text):
        # 打印输入张量的形状
        print(f"motion_shape: {motion.shape}, text_shape: {text.shape}")

        motion_features = self.motion_encoder(motion)
        print("motion_features_shape", motion_features.shape)
        text_features = self.text_encoder(text)
        print("text_features_shape", text_features.shape)

        combined_features = torch.cat((motion_features, text_features), dim=2)
        print("cat_shape", combined_features.shape)

        # # 对每个段进行平均池化
        # # clip level
        # combined_features = combined_features.mean(dim=1)
        # print("cat_mean_shape_clip_level", combined_features.shape)


        # # 将特征展平为 (batch_size * seq_len, feature_dim)
        # # frame level
        # combined_features = combined_features.view(-1, 512)  # 调整维度为512
        # print("cat_view_shape_frame_level", combined_features.shape)

        # 不进行平均池化，保持帧级别
        combined_features = combined_features.view(-1, 512)  # 保持帧级别
        print("combined_features_shape_frame_level", combined_features.shape)

        output = self.fc(combined_features)
        return output


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        self.threshold = args.threshold

        # self.rank = dist.get_rank()
        self.ori_stride = self.args.stride
        self.ori_length = self.args.pose_length
        self.alignment = [0,0] # for trinity

        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list = joints_list[self.args.tar_joints]
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys())) + 1
            self.joint_mask = np.zeros(self.joints * 3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # select trainable joints
        self.smplx = smplx.create(
            self.args.smplx_model_path,
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).cuda().eval()

        split_rule = pd.read_csv(args.data_path + "\\train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]
        self.data_dir = args.data_path

        if loader_type == "test":
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])

        # self.lang_model = None
        if args.word_rep is not None:
            with open(f"{args.data_path}\\weights\\vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)

        # For Train Val Test g_cache respectively
        args.preloaded_dir += loader_type
        
        if build_cache:
            self.build_cache(args.preloaded_dir)
        self.lmdb_env = lmdb.open(args.preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]

    def build_cache(self, preloaded_dir):
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True,
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering,
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)

    def __len__(self):
        return self.n_samples
    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir):
            os.makedirs(out_lmdb_dir)
        # if len(self.args.training_speakers) == 1:
        #     dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 50))# 50G
        # else:
        #     dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 200))# 200G
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 50)) if len(self.args.training_speakers) == 1 else lmdb.open(out_lmdb_dir, map_size=int(1024 ** 3 * 200))
        n_filtered_out = defaultdict(int)

        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir  + "\\" + self.args.pose_rep + "\\" + f_name + ext
            pose_each_file = []
            trans_each_file = []
            shape_each_file = []

            word_each_file = []

            sem_each_file = []

            id_pose = f_name #1_wayne_0_1_1

            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30 % self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30 / self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride]
                trans_each_file = pose_data["trans"][::stride]
                shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)

                assert self.args.pose_fps == 30, "should 30"
                m_data = np.load(pose_file, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length],
                            transl=trans[i*max_length:(i+1)*max_length],
                            expression=exps[i*max_length:(i+1)*max_length],
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69],
                            global_orient=poses[i*max_length:(i+1)*max_length,:3],
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3],
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3],
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72],
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r],
                            transl=trans[s*max_length:s*max_length+r],
                            expression=exps[s*max_length:s*max_length+r],
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69],
                            global_orient=poses[s*max_length:s*max_length+r,:3],
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3],
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3],
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3],
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72],
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0) # all, 4, 3
                # print(joints.shape)
                feetv = torch.zeros(joints.shape[1], joints.shape[0])
                joints = joints.permute(1, 0, 2)
                #print(joints.shape, feetv.shape)
                feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
                #print(feetv.shape)
                contacts = (feetv < 0.01).numpy().astype(float)
                # print(contacts.shape, contacts)
                contacts = contacts.transpose(1, 0)
                pose_each_file = pose_each_file * self.joint_mask
                pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
                # print(pose_each_file.shape)


            else:
                assert 120 % self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                stride = int(120 / self.args.pose_fps)
                with open(pose_file, "r") as pose_data:
                    for j, line in enumerate(pose_data.readlines()):
                        if j < 431:
                            continue
                        if j % stride != 0:
                            continue
                        data = np.fromstring(line, dtype=float, sep=" ")
                        rot_data = rc.euler_angles_to_matrix(torch.from_numpy(np.deg2rad(data)).reshape(-1, self.joints, 3), "XYZ")
                        rot_data = rc.matrix_to_axis_angle(rot_data).reshape(-1, self.joints * 3)
                        rot_data = rot_data.numpy() * self.joint_mask

                        pose_each_file.append(rot_data)
                        trans_each_file.append(data[:3])

                pose_each_file = np.array(pose_each_file)
                # print(pose_each_file.shape)
                trans_each_file = np.array(trans_each_file)
                shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)





            time_offset = 0
            if self.args.word_rep is not None:
                logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                word_file = f"{self.data_dir}\\{self.args.word_rep}\\{id_pose}.TextGrid"
                if not os.path.exists(word_file):
                    logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                tgrid = tg.TextGrid.fromFile(word_file)
                if self.args.t_pre_encoder == "bert":
                    from transformers import AutoTokenizer, BertModel
                    tokenizer = AutoTokenizer.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True)
                    model = BertModel.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True).eval()
                    list_word = []
                    all_hidden = []
                    max_len = 400
                    last = 0
                    word_token_mapping = []
                    first = True
                    for i, word in enumerate(tgrid[0]):
                        last = i
                        if (i%max_len != 0) or (i==0):
                            if word.mark == "":
                                list_word.append(".")
                            else:
                                list_word.append(word.mark)
                        else:
                            max_counter = max_len
                            str_word = ' '.join(map(str, list_word))
                            if first:
                                global_len = 0
                            end = -1
                            offset_word = []
                            for k, wordvalue in enumerate(list_word):
                                start = end+1
                                end = start+len(wordvalue)
                                offset_word.append((start, end))
                            #print(offset_word)
                            token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                            #print(token_scan)
                            for start, end in offset_word:
                                sub_mapping = []
                                for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                    if int(start) <= int(start_t) and int(end_t) <= int(end):
                                        #print(i+global_len)
                                        sub_mapping.append(i+global_len)
                                word_token_mapping.append(sub_mapping)
                            #print(len(word_token_mapping))
                            global_len = word_token_mapping[-1][-1] + 1
                            list_word = []
                            if word.mark == "":
                                list_word.append(".")
                            else:
                                list_word.append(word.mark)

                            with torch.no_grad():
                                inputs = tokenizer(str_word, return_tensors="pt")
                                outputs = model(**inputs)
                                last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                            all_hidden.append(last_hidden_states)

                    #list_word = list_word[:10]
                    if list_word == []:
                        pass
                    else:
                        if first:
                            global_len = 0
                        str_word = ' '.join(map(str, list_word))
                        end = -1
                        offset_word = []
                        for k, wordvalue in enumerate(list_word):
                            start = end+1
                            end = start+len(wordvalue)
                            offset_word.append((start, end))
                        #print(offset_word)
                        token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                        #print(token_scan)
                        for start, end in offset_word:
                            sub_mapping = []
                            for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                if int(start) <= int(start_t) and int(end_t) <= int(end):
                                    sub_mapping.append(i+global_len)
                                    #print(sub_mapping)
                            word_token_mapping.append(sub_mapping)
                        #print(len(word_token_mapping))
                        with torch.no_grad():
                            inputs = tokenizer(str_word, return_tensors="pt")
                            outputs = model(**inputs)
                            last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                        all_hidden.append(last_hidden_states)
                    last_hidden_states = np.concatenate(all_hidden, axis=0)

            if os.path.exists(word_file):
                tgrid = tg.TextGrid.fromFile(word_file)
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    current_time = i/self.args.pose_fps + time_offset
                    j_last = 0
                    for j, word in enumerate(tgrid[0]):
                        word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                        if word_s<=current_time and current_time<=word_e:
                            if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                                mapping_index = word_token_mapping[j]
                                #print(mapping_index, word_s, word_e)
                                s_t = np.linspace(word_s, word_e, len(mapping_index)+1)
                                #print(s_t)
                                for tt, t_sep in enumerate(s_t[1:]):
                                    if current_time <= t_sep:
                                        #if len(mapping_index) > 1: print(mapping_index[tt])
                                        word_each_file.append(last_hidden_states[mapping_index[tt]])
                                        break
                            else:
                                if word_n == " ":
                                    word_each_file.append(self.lang_model.PAD_token)
                                else:
                                    word_each_file.append(self.lang_model.get_word_index(word_n))
                            found_flag = True
                            j_last = j
                            break
                        else: continue
                    if not found_flag:
                        if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                            word_each_file.append(last_hidden_states[j_last])
                        else:
                            word_each_file.append(self.lang_model.UNK_token)
                word_each_file = np.array(word_each_file)
                #print(word_each_file.shape)



            # TODO:
            #
            if self.args.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = f"{self.data_dir}\\{self.args.sem_rep}\\{id_pose}.txt"
                sem_all = pd.read_csv(sem_file,
                    sep='\t',
                    names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                
                # we adopt motion-level semantic score here.
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.args.pose_fps + time_offset
                        if start<=current_time and current_time<=end:
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)

            filtered_result = self._sample_from_clip(dst_lmdb_env,
                                                     pose_each_file,
                                                     trans_each_file,
                                                     shape_each_file,
                                                     word_each_file,

                                                     sem_each_file,
                                                     disable_filtering,
                                                    clean_first_seconds,
                                                    clean_final_seconds,
                                                    is_test)
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]

        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def _sample_from_clip(
        self,
        dst_lmdb_env,
        pose_each_file,
        trans_each_file,
        shape_each_file,
        word_each_file,

        sem_each_file,
        disable_filtering,
        clean_first_seconds,
        clean_final_seconds,
        is_test
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data
        """

        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # assume 1500 frames / 30 fps = 50 s
        #print(round_seconds_skeleton)
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [0, 50]s

        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [0,50*30]


        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)

            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")


            n_filtered_out = defaultdict(int)
            sample_pose_list = []



            sample_shape_list = []
            sample_word_list = []

            sample_sem_list = []

            sample_trans_list = []

            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)

                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])

                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])


                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)


                        sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)
                    else:
                        n_filtered_out[filtering_message] += 1

            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, shape, word,  sem, trans in zip(
                        sample_pose_list,
                        sample_shape_list,
                        sample_word_list,

                        sample_sem_list,
                        sample_trans_list,):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")

                        # 序列化&反序列化
                        v = [pose, shape, word,  sem, trans]
                        v = pyarrow.serialize(v).to_buffer()
                        # v = pickle.dumps([pose, shape, word, sem, trans])
                        txn.put(k, v)
                        self.n_out_samples += 1
        return n_filtered_out

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            # 序列化&反序列化
            sample = pyarrow.deserialize(sample)
            # sample = pickle.loads(sample)
            tar_pose, in_shape, in_word, sem, trans = sample
            # tar_pose, trans, in_shape, in_word, sem = sample
            sem = torch.from_numpy(np.copy(sem)).float()
            in_word = torch.from_numpy(np.copy(in_word)).float() if self.args.word_cache else torch.from_numpy(np.copy(in_word)).int()


            # print(f"word shape: {in_word.shape}, word: {in_word}")
            # print(f"sem shape: {sem.shape}, sem: {sem}")
            # print(f"word shape: {in_word.shape}")
            # print(f"sem shape: {sem.shape}")

            vocab_size = self.lang_model.n_words
            if torch.max(in_word) >= vocab_size or torch.min(in_word) < 0:
                print(f"__getitem__ tensor out of bounds")
                print(f"Warning: Word indices out of bounds before processing. Max index: {torch.max(in_word)}, Min index: {torch.min(in_word)}, Vocab size: {vocab_size}")
                in_word = torch.clamp(in_word, min=0, max=vocab_size-1)

            # 打印调试信息
            # print(f"__getitem__ - in_word max: {torch.max(in_word)}, min: {torch.min(in_word)}, vocab_size: {vocab_size}")



            if self.loader_type == "test":
                tar_pose = torch.from_numpy(np.copy(tar_pose)).float()
                trans = torch.from_numpy(np.copy(trans)).float()
                # in_facial = torch.from_numpy(in_facial).float()

                in_shape = torch.from_numpy(np.copy(in_shape)).float()
                # #### TODO clip level - mean 0.188(average semantic score)

                # label = 1 if sem.mean().item() > 0.188 else 0

                ### TODO Frame-level label
                # 生成帧级别的标签
                labels = torch.zeros_like(sem, dtype=torch.long)
                labels[sem > 0.1] = 1

                ### TODO clip level - frame count threshold

                # # 计算大于 0.1 的帧数
                # count_above_threshold = torch.sum(sem > 0.1).item()
                # label = 1 if count_above_threshold > (sem.shape[0] * self.threshold) else 0


            else:
                in_shape = torch.from_numpy(np.copy(in_shape)).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(np.copy(trans)).reshape((trans.shape[0], -1)).float()

                tar_pose = torch.from_numpy(np.copy(tar_pose)).reshape((tar_pose.shape[0], -1)).float()
                # in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
                # label = 2 if sem.mean().item() > 0.1 else (1 if sem.mean().item() > 0 else 0)
                # print(f"beta shape: {in_shape.shape}")
                # print(f"trans shape: {trans.shape}")
                # print(f"pose shape: {tar_pose.shape}")


                # #### TODO clip level - mean 0.188(average semantic score)

                # label = 1 if sem.mean().item() > 0.188 else 0

                ### TODO Frame-level label
                # 生成帧级别的标签
                labels = torch.zeros_like(sem, dtype=torch.long)
                labels[sem > 0.1] = 1

                ### TODO clip level - frame count threshold

                # # 计算大于 0.1 的帧数
                # count_above_threshold = torch.sum(sem > 0.1).item()
                # label = 1 if count_above_threshold > (sem.shape[0] * self.threshold) else 0


            # # 将 label 转换为张量
            # label = torch.tensor(labels, dtype=torch.long)
            # # print("label shape",label.shape)







            return {"pose": tar_pose, "beta": in_shape, "word": in_word, "sem": sem, "trans": trans, "label": labels}

class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False

'''
def collate_fn(batch):
    # 直接将 'pose'、'beta'、'word'、'sem'、'trans' 和 'label' 保留为列表形式
    poses = [item['pose'] for item in batch]
    betas = [item['beta'] for item in batch]
    words = [item['word'] for item in batch]
    sems = [item['sem'] for item in batch]
    trans = [item['trans'] for item in batch]
    labels = [item['label'] for item in batch]

    # poses_padded = pad_sequence(poses, batch_first=True)
    # betas_padded = pad_sequence(betas, batch_first=True)
    # words_padded = pad_sequence(trans, batch_first=True)
    # sems_padded = pad_sequence(sems, batch_first=True)
    # trans_padded = pad_sequence(trans, batch_first=True)

    poses_tensor = torch.tensor(poses, dtype=torch.long)
    betas_tensor = torch.tensor(betas, dtype=torch.long)
    words_tensor = torch.tensor(words, dtype=torch.long)
    sems_tensor = torch.tensor(sems, dtype=torch.long)
    trans_tensor = torch.tensor(trans, dtype=torch.long)


    #clip-based
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # # frame-based
    # labels_padded = pad_sequence(labels, batch_first=True)

    return {
        "pose": poses_tensor,
        "beta": betas_tensor,
        "word": words_tensor,
        "sem": sems_tensor,
        "trans": trans_tensor,
        "label": labels_tensor
    }
'''
'''
def collate_fn(batch):
    # 保留 'pose'、'beta'、'word'、'sem'、'trans' 和 'label' 为列表形式
    poses = [item['pose'] for item in batch]
    betas = [item['beta'] for item in batch]
    words = [item['word'] for item in batch]
    sems = [item['sem'] for item in batch]
    trans = [item['trans'] for item in batch]
    labels = [item['label'] for item in batch]

    return {
        "pose": poses,
        "beta": betas,
        "word": words,
        "sem": sems,
        "trans": trans,
        "label": labels
    }
'''

def custom_collate_fn(batch):
    pose_batch = torch.stack([item['pose'] for item in batch])
    beta_batch = torch.stack([item['beta'] for item in batch])
    word_batch = torch.stack([item['word'] for item in batch])
    sem_batch = torch.stack([item['sem'] for item in batch])
    trans_batch = torch.stack([item['trans'] for item in batch])

    # clip-based
    # label_batch = torch.tensor([item['label'].item() for item in batch])

    # frame—based
    label_batch = torch.stack([item['label'] for item in batch])

    # # 打印调试信息
    # print(f"Collate fn - Pose batch shape: {pose_batch.shape}")
    # print(f"Collate fn - Beta batch shape: {beta_batch.shape}")
    # print(f"Collate fn - Word batch shape: {word_batch.shape}")
    # print(f"Collate fn - Sem batch shape: {sem_batch.shape}")
    # print(f"Collate fn - Trans batch shape: {trans_batch.shape}")
    # print(f"Collate fn - Label batch shape: {label_batch.shape}")

    # 打印 word_batch 的详细信息
    # print(f"Collate fn - Word batch max: {torch.max(word_batch)}, min: {torch.min(word_batch)}")

    # 确保 word 中的索引在词汇表范围内
    # with open(f"{args.data_path}\\weights\\vocab.pkl", 'rb') as f:
    #             lang_model = pickle.load(f)
    vocab_size = 11195
    if torch.max(word_batch) >= vocab_size or torch.min(word_batch) < 0:
        print(f"Collate fn - Word batch out of bounds. Max index: {torch.max(word_batch)}, Min index: {torch.min(word_batch)}, Vocab size: {vocab_size}")
        raise ValueError(f"Word batch indices are out of bounds. Max index: {torch.max(word_batch)}, Min index: {torch.min(word_batch)}, Vocab size: {vocab_size}")

    return {
        "pose": pose_batch,
        "beta": beta_batch,
        "word": word_batch,
        "sem": sem_batch,
        "trans": trans_batch,
        "label": label_batch
    }

# 全局变量来存储标签计数
global_label_counts = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

# tracker = other_tools.EpochTracker(["loss", "accuracy", "precision", "recall", "f1"], [False, True, True, True, True])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output_path, log_path, num_classes):
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
    best_val_acc = 0.0
    tracker = other_tools.EpochTracker(["loss", "accuracy", "precision", "recall", "f1"], [False, True, True, True, True])

    for epoch in range(num_epochs):
        model.train()
        tracker.reset()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        global_label_counts["train"].clear()

        for batch in train_loader:
            try:
                # 重新初始化 CUDA 设备
                torch.cuda.empty_cache()

                # 从 batch 中提取数据，并移动到 GPU
                inputs = batch['pose'].cuda()
                betas = batch['beta'].cuda()
                words = batch['word'].cuda().long()  # 确保转换为 long 类型
                sems = batch['sem'].cuda()
                trans = batch['trans'].cuda()
                labels = batch['label'].cuda()

                print(f"Training batch labels: {labels}")
                print(f"Max label: {labels.max()}, Min label: {labels.min()}")

                optimizer.zero_grad()
                outputs = model(inputs, words)

                print(f"[Training Before View]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Training Before View]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                outputs = outputs.view(-1, num_classes)  # 展平为 (batch_size * seq_len, num_classes)
                labels = labels.view(-1)  # 展平为 (batch_size * seq_len)

                print(f"[Training]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Training]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                tracker.update_meter("loss", "train", loss.item())

                for label in labels.cpu().numpy():
                    global_label_counts["train"][label] += 1

            except Exception as e:
                print(f"Error during training: {e}")
                torch.cuda.synchronize()  # 强制同步 CUDA 设备
                raise  # 重新引发异常，停止执行

        train_acc = accuracy_score(all_labels, all_preds)
        tracker.update_meter("accuracy", "train", train_acc)

        val_metrics = evaluate_model(model, val_loader, criterion, num_classes, output_path, epoch)
        val_acc, val_loss, val_precision, val_recall, val_f1 = val_metrics

        tracker.update_meter("loss", "val", val_loss)
        tracker.update_meter("accuracy", "val", val_acc)
        tracker.update_meter("precision", "val", val_precision)
        tracker.update_meter("recall", "val", val_recall)
        tracker.update_meter("f1", "val", val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            torch.save(model.state_dict(), os.path.join(output_path, 'best_motion_text_classifier.pth'))

        log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}"
        print(log_message)
        logging.info(log_message)
        print(f"Train Label counts: {dict(global_label_counts['train'])}")
        logging.info(f"Train Label counts: {dict(global_label_counts['train'])}")
        print(f"Validation Label counts: {dict(global_label_counts['val'])}")
        logging.info(f"Validation Label counts: {dict(global_label_counts['val'])}")

def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def log_misclassification(misclassified_samples, output_path):
    log_file = os.path.join(output_path, 'misclassification_log.txt')
    with open(log_file, 'w') as f:
        for true, pred in misclassified_samples:
            f.write(f"True: {true}, Pred: {pred}\n")

def evaluate_model(model, dataloader, criterion, num_classes, output_path, epoch):
    model.eval()
    tracker = other_tools.EpochTracker(["loss", "accuracy", "precision", "recall", "f1"], [False, True, True, True, True])
    tracker.reset()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    global_label_counts["val"].clear()

    with torch.no_grad():
        for batch in dataloader:
            try:
                # 重新初始化 CUDA 设备
                torch.cuda.empty_cache()

                # 从 batch 中提取数据，并移动到 GPU
                inputs = batch['pose'].cuda()
                betas = batch['beta'].cuda()
                words = batch['word'].cuda()
                sems = batch['sem'].cuda()
                trans = batch['trans'].cuda()
                labels = batch['label'].cuda()

                print(f"Validation batch labels: {labels}")
                print(f"Max label: {labels.max()}, Min label: {labels.min()}")

                outputs = model(inputs, words)

                print(f"[Validation Before View]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Validation Before View]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                outputs = outputs.view(-1, num_classes)  # 展平为 (batch_size * seq_len, num_classes)
                labels = labels.view(-1)  # 展平为 (batch_size * seq_len)

                loss = criterion(outputs, labels)

                print(f"[Validation]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Validation]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                tracker.update_meter("loss", "val", loss.item())

                for label in labels.cpu().numpy():
                    global_label_counts["val"][label] += 1

            except Exception as e:
                print(f"Error during validation: {e}")
                torch.cuda.synchronize()  # 强制同步 CUDA 设备
                raise  # 重新引发异常，停止执行

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    tracker.update_meter("accuracy", "val", acc)
    tracker.update_meter("precision", "val", precision)
    tracker.update_meter("recall", "val", recall)
    tracker.update_meter("f1", "val", f1)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(output_path, f'confusion_matrix_epoch_{epoch}.png')
    plot_confusion_matrix(cm, cm_path)

    # Log misclassification details
    misclassified_samples = [(true, pred) for true, pred in zip(all_labels, all_preds) if true != pred]
    log_misclassification(misclassified_samples, output_path)

    log_message = f"Validation - Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}, Acc: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
    print(log_message)
    logging.info(log_message)
    print(f"Validation Label counts: {dict(global_label_counts['val'])}")
    logging.info(f"Validation Label counts: {dict(global_label_counts['val'])}")

    return acc, running_loss / len(dataloader), precision, recall, f1

def test_model(model, dataloader, criterion, num_classes):
    model.eval()
    tracker = other_tools.EpochTracker(["loss", "accuracy", "precision", "recall", "f1"], [False, True, True, True, True])
    tracker.reset()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    global_label_counts["test"].clear()

    with torch.no_grad():
        for batch in dataloader:
            try:
                torch.cuda.empty_cache()

                inputs = batch['pose'].cuda()  # 形状 (1, seq_len, 169)
                betas = batch['beta'].cuda()
                words = batch['word'].cuda()  # 形状 (1, seq_len)
                sems = batch['sem'].cuda()
                trans = batch['trans'].cuda()
                labels = batch['label'].cuda()  # 形状 (1, seq_len)

                outputs = model(inputs, words)  # 输出形状为 (1, seq_len, num_classes)

                print(f"[Test Before View]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Test Before View]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                outputs = outputs.view(-1, num_classes)  # 展平为 (seq_len, num_classes)
                labels = labels.view(-1)  # 展平为 (seq_len)

                loss = criterion(outputs, labels)

                print(f"[Test]Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
                print(f"[Test]Outputs sample: {outputs[0]}, Labels sample: {labels[0]}")

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

                tracker.update_meter("loss", "test", loss.item())

                for label in labels.cpu().numpy():
                    global_label_counts["test"][label] += 1

            except Exception as e:
                print(f"Error during testing: {e}")
                torch.cuda.synchronize()
                raise

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    tracker.update_meter("accuracy", "test", acc)
    tracker.update_meter("precision", "test", precision)
    tracker.update_meter("recall", "test", recall)
    tracker.update_meter("f1", "test", f1)

    log_message = f"Test - Loss: {running_loss/len(dataloader)}, Acc: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
    print(log_message)
    logging.info(log_message)
    print(f"Test Label counts: {dict(global_label_counts['test'])}")
    logging.info(f"Test Label counts: {dict(global_label_counts['test'])}")

    return acc, running_loss / len(dataloader), precision, recall, f1

def main(output_path, log_path):
    start_time = time.time()

    class Args:
        stride = 30
        pose_length = 128
        pose_fps = 30
        data_path = "D:\\MingYuan\\Dataset\\beat_v2.0.0\\beat_english_v2.0.0"
        smplx_model_path = "D:\\MingYuan\\smplx_v1.1\\models"
        additional_data = False
        loader_type = "train"
        preloaded_dir = ".\\datasets_cache"
        training_speakers = [2]
        pose_rep = "smplxflame_30"
        ori_joints = "beat_smplx_joints"
        tar_joints = "beat_smplx_full"
        sem_rep = "sem"
        facial_rep = None
        new_cache = False
        disable_filtering = False
        multi_length_training = [1.0]
        word_rep = "textgrid"
        clean_first_seconds = 0
        clean_final_seconds = 0
        t_pre_encoder = "fasttext"
        word_cache = False
        threshold = 0.5

    args = Args()

    dataset_start_time = time.time()
    train_dataset = CustomDataset(args, "train", build_cache=True)
    val_dataset = CustomDataset(args, "val", build_cache=False)
    test_dataset = CustomDataset(args, "test", build_cache=False)
    dataset_end_time = time.time()

    data_loader_start_time = time.time()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)  # batch_size = 1 for test
    data_loader_end_time = time.time()

    motion_input_dim = 169
    text_input_dim = 11195
    embed_dim = 300
    text_output_dim = 256
    hidden_size = 512
    num_classes = 2

    model = MotionTextClassifier(motion_input_dim, text_input_dim, embed_dim, text_output_dim, hidden_size, num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_start_time = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, output_path=output_path, log_path=log_path, num_classes=num_classes)
    train_end_time = time.time()

    # Evaluate model on validation set
    val_metrics = evaluate_model(model, val_loader, criterion, num_classes, output_path, epoch=0) # 增加 epoch 参数
    val_acc, val_loss, val_precision, val_recall, val_f1 = val_metrics
    print(f"Validation Loss: {val_loss}, Validation Acc: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}")

    # Test model
    test_start_time = time.time()
    test_metrics = test_model(model, test_loader, criterion, num_classes)
    test_acc, test_loss, test_precision, test_recall, test_f1 = test_metrics
    test_end_time = time.time()
    print(f"Test Loss: {test_loss}, Test Acc: {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1}")

    end_time = time.time()

    total_time = end_time - start_time
    dataset_time = dataset_end_time - dataset_start_time
    data_loader_time = data_loader_end_time - data_loader_start_time
    training_time = train_end_time - train_start_time
    testing_time = test_end_time - test_start_time

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Dataset preparation time: {dataset_time:.2f} seconds")
    print(f"DataLoader preparation time: {data_loader_time:.2f} seconds")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Testing time: {testing_time:.2f} seconds")

if __name__ == "__main__":
    root = "D:\\MingYuan\\A2M"
    output_path = os.path.join(root, "classifier")
    log_path = os.path.join(output_path, "training_log.txt")
    main(output_path, log_path)