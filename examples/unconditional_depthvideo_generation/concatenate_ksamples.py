import os
import glob
import cv2
from pathlib import Path
import numpy as np
import random

if __name__ == "__main__":

    instance_data_root = Path("/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_sampling-aidedautoregressive/")
    ext = ['png']
    videos = ['Charades/7177T', 'Charades/81YUE', 'Charades/9M48H', 'Charades/9UU4W', 'Charades/PJUM0', 'AVA/YAAUPjq-L-Q_scene_0_36558-37301', 'HACS/Mowing_the_lawn_v_rin4oz7Fego_scene_0_0-1696', 'HACS/Making_a_lemonade_v_Zyo70ZiXmYY_scene_0_1938-3070', 'HACS/Painting_furniture_v_xNxxM-OOMfw_scene_0_0-1910', 'HACS/Raking_leaves_v_iSjk42F0rvM']
    num_rows = 1
    num_cols = 5
    num_frames = 13
    count = 0
    width, height = num_cols * 64, num_rows * 64

    for vid in videos:
        all_frames = []
        seq_to_frames = {}
        blank_image = [np.zeros((height, width, 3)) for i in range(num_frames)]
        vid_frames = []
        vid_data_root = str(instance_data_root / f"{vid}_1_depth_nogt")

        print(vid_data_root)

        for e in ext:
            all_frames.extend(sorted(list(glob.glob(f"{vid_data_root}/**/00*.{e}", recursive=True))))
        all_frames = list(all_frames)

        print("number of frames found", len(all_frames))

        for frame in all_frames:
            seq = str(frame)[len(str(instance_data_root)):str(frame).rfind("/")]
            if seq not in seq_to_frames:
                seq_to_frames[seq] = []
            seq_to_frames[seq].append(frame)

        print("found total videos to be ", len(list(seq_to_frames.keys())))
        dkeys = list(seq_to_frames.keys())[:num_rows*num_cols]
        random.shuffle(dkeys)

        for i in range(num_frames):
            bi = blank_image[i]
            for j, seq in enumerate(dkeys):
                cell = (int(j / num_cols), int(j % num_cols))
                if i > len(seq_to_frames[seq]) - 1:
                    print(seq)
                    frame = np.ones((64, 64, 3)) * 255
                else:
                    frame = cv2.imread(str(seq_to_frames[seq][i]))
                bi[cell[0] * 64: (cell[0] + 1) * 64, cell[1] * 64: (cell[1] + 1) * 64] = frame
            vid_frames.append(bi)


        fcc = cv2.VideoWriter_fourcc(*'MP4V')
        vidfile = vid.replace('/', '_')
        video = cv2.VideoWriter(os.path.join(f"/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_sampling-aidedautoregressive/{vidfile}_ksamples.mp4"), fcc, 1.0, (width,height))
        for frame in vid_frames:
            video.write(frame.astype(np.uint8))
        video.release()
