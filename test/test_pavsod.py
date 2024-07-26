import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import argparse
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import numpy as np
import cv2

import torch
from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from pavsodr_model import add_model_config_soundnet
from predictor import VisualizationDemo

from evaluator import Eval_thread
from dataloader import EvalDataset

import torchaudio

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_model_config_soundnet(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="model test configs")
    parser.add_argument(
        "--config-file",
        default="./configs/pavsodr/R50_PAVSOD.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--model_input",
        default="./pre_models/"
    )
    parser.add_argument(
        "--image_input",
        default="./datasets/pavsodr/test/JPEGImages/"
    )
    parser.add_argument(
        "--audio_input",
        default = "./datasets/pavsodr/audios/"
    )
    parser.add_argument(
        "--output",
        default="./output_pavsod/results/"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=2,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    pths = ["model_pavsod.pth"]

    for f in pths:
        print(f)
        cfg["MODEL"]["WEIGHTS"] = args.model_input + f

        demo = VisualizationDemo(cfg)

        output_path = os.path.join(args.output, f.split(".")[0])

        if output_path:
            os.makedirs(output_path, exist_ok=True)

        if args.image_input:
            num_frames = args.num_frames
            for video in os.listdir(args.image_input):
                all_frames = os.listdir(os.path.join(args.image_input, video))
                sorted_all_frames = sorted(all_frames, key=lambda x: int(x.split('.')[0]))

                audio_pth = args.audio_input + video.split("/")[0] + '.wav'
                # print(audio_pth)
                audio_ori = torchaudio.load(audio_pth)[0]

                video_length = len(sorted_all_frames)
                audio_ori_ch1 = audio_ori[0]
                audio_ori_ch2 = audio_ori[1]
                audio_ori_ch3 = audio_ori[2]
                audio_ori_ch4 = audio_ori[3]
                audio_split_size = int(len(audio_ori[1]) / (int(video_length)))
                audio_ch1_split = torch.split(tensor=audio_ori_ch1, split_size_or_sections=audio_split_size)
                audio_ch1_split = list(audio_ch1_split)
                audio_ch1_split = audio_ch1_split[:int(video_length)]
                audio_ch2_split = torch.split(tensor=audio_ori_ch2, split_size_or_sections=audio_split_size)
                audio_ch2_split = list(audio_ch2_split)
                audio_ch2_split = audio_ch2_split[:int(video_length)]
                audio_ch3_split = torch.split(tensor=audio_ori_ch3, split_size_or_sections=audio_split_size)
                audio_ch3_split = list(audio_ch3_split)
                audio_ch3_split = audio_ch3_split[:int(video_length)]
                audio_ch4_split = torch.split(tensor=audio_ori_ch4, split_size_or_sections=audio_split_size)
                audio_ch4_split = list(audio_ch4_split)
                audio_ch4_split = audio_ch4_split[:int(video_length)]

                for i in range(0, len(sorted_all_frames), num_frames):
                    clip_frames = sorted_all_frames[i:i + num_frames]
                    vid_frames = []
                    vid_frames_name = []
                    aud_frames = []

                    for img_name in clip_frames:
                        vid_frames_name.append(img_name)
                        img = read_image(os.path.join(args.image_input, video, img_name), format="BGR")
                        vid_frames.append(img)

                    # audio:
                    frame_idx = int(clip_frames[0].split(".")[0])

                    if frame_idx in [0, 1, 2, 3, 4]:
                        start_index = 0
                        end_index = 11
                    elif frame_idx in [video_length-5, video_length-4, video_length-3, video_length-2, video_length-1]:
                        start_index = video_length - 11
                        end_index = video_length
                    else:
                        start_index = frame_idx - 5
                        end_index = frame_idx + 6

                    assert end_index - start_index == 11

                    audios = []
                    a_ch1 = audio_ch1_split[start_index: end_index]
                    a_ch2 = audio_ch2_split[start_index: end_index]
                    a_ch3 = audio_ch3_split[start_index: end_index]
                    a_ch4 = audio_ch4_split[start_index: end_index]
                    audios.append([torch.cat(a_ch1), torch.cat(a_ch2), torch.cat(a_ch3), torch.cat(a_ch4)])
                    audios = torch.stack(audios[0], dim=0)
                    audios = torch.mean(audios, dim=0).unsqueeze(0).unsqueeze(0)

                    aud_frames.append(audios)

                    add_imgs_count = len(vid_frames) % num_frames
                    with autocast():
                        if add_imgs_count == 0:
                            binary_masks = demo.run_on_video(vid_frames, aud_frames, args.confidence_threshold)
                        else:
                            for i in range(num_frames - len(vid_frames) % num_frames):
                                vid_frames.append(np.zeros_like(vid_frames[0]))
                            binary_masks = demo.run_on_video(vid_frames, aud_frames, args.confidence_threshold)

                    # binary images
                    if add_imgs_count == 0:
                        for i in range(len(binary_masks)):
                            if not os.path.exists(os.path.join(output_path, video)):
                                os.makedirs(os.path.join(output_path, video))
                            cv2.imwrite(os.path.join(output_path, video, vid_frames_name[i]), binary_masks[i])
                    else:
                        for i in range(add_imgs_count):
                            if not os.path.exists(os.path.join(output_path, video)):
                                os.makedirs(os.path.join(output_path, video))
                            cv2.imwrite(os.path.join(output_path, video, vid_frames_name[i]), binary_masks[i])


        gt_dir = "./datasets/pavsodr/test/Annotations/"

        threads = []
        loader = EvalDataset(output_path, gt_dir)
        thread = Eval_thread(loader, method="", dataset="", output_dir=output_path, cuda=True)
        threads.append(thread)

        for thread in threads:
            print(thread.run())
