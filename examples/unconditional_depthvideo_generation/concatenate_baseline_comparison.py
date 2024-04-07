import numpy as np
import cv2
from pathlib import Path
import glob
import os
import random
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


if __name__ == "__main__":

    instance_data_root = Path("/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_visuals_ddpm/")
    ext = ['png']
    all_frames = []
    seq_to_frames = {}
    count = 0
    width, height = 10 * 64, 64
    vid_frames = []

    cmap = matplotlib.cm.get_cmap('jet_r')
    cmap = truncate_colormap(cmap, 0.6, 1.0)

    for e in ext:
        all_frames.extend(sorted(list(instance_data_root.rglob(f"*.{e}"))))
    all_frames = list(all_frames)

    for frame in all_frames:
        seq = str(frame)[len(str(instance_data_root)):str(frame).rfind("/")]
        if seq not in seq_to_frames:
            seq_to_frames[seq] = []
        seq_to_frames[seq].append(str(frame))

    print("found total videos to be ", len(list(seq_to_frames.keys())))
    dkeys = list(seq_to_frames.keys())

    for i, seq in enumerate(dkeys):
        print(seq_to_frames[seq])
        ours = cv2.imread(seq_to_frames[seq][0])
        gt = cv2.imread(seq_to_frames[seq][1])
        input1 = cv2.imread(seq_to_frames[seq][2])
        input2 = cv2.imread(seq_to_frames[seq][3])
        input3 = cv2.imread(seq_to_frames[seq][4])
        constant_past = input3.copy()
        constant_past = 1 - (constant_past - constant_past.min()) / (constant_past.max() - constant_past.min())
        constant_past = (cmap(rgb2gray(constant_past)) * 255).astype(np.uint8)[..., :3][..., ::-1]
        # extrapolation = cv2.imread(seq_to_frames[seq][0].replace('train_visuals_ddpm', 'train_visuals_interpolation'))
        regression = cv2.imread(seq_to_frames[seq][0].replace('TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000', 'TAO-depth_val_regressionbaseline_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_2shorizon_startfromSD_finetuneverything/checkpoint-5000'))
        mcvd = cv2.imread(seq_to_frames[seq][0].replace('TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000', 'TAO-depth_val_img2img_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_2shorizon_startfromSD_finetuneverything/checkpoint-5000'))
        river = cv2.imread(seq_to_frames[seq][0].replace('TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_visuals_ddpm', 'river-results-customvqgan').replace('depth.png', 'output.png'))
        fdm = cv2.imread(seq_to_frames[seq][0].replace('TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_visuals_ddpm', 'fdm-results-1s').replace('depth.png', 'output.png'))
        vid_frame = np.hstack([input1, input2, input3, constant_past, regression, river, mcvd, fdm, ours, gt])

        # put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 10)
        fontScale = 0.4
        color = (0, 0, 0)
        thickness = 1
        # vid_frame = cv2.putText(vid_frame, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)

        vid_frames.append(vid_frame)

    supp_figure = []

    for i, frame in enumerate(vid_frames):
        if i in [2,8,11,14,31,37,44,49,60,64,66,79,84,85,88,93,104,119,127,149,163,178,202,212,217,236,244,253,261,262,264,275,277,280,282,290,294,303,306,312,320,322,330,339,343,355,357,421,464,468,491,496
                ]:
            supp_figure.append(frame)

    random.shuffle(supp_figure)
    supp_figure = np.vstack(supp_figure)

    cv2.imwrite('/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/baseline_comparison.png', supp_figure)

    """
    fcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(os.path.join("/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_visuals_ddpm/baseline_comparison.mp4"), fcc, 0.25, (width,height))
    for frame in vid_frames:
        video.write(frame.astype(np.uint8))
    video.release()
    """
