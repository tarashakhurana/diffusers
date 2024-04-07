import os
import cv2
import numpy as np

if __name__ == "__main__":

    ########################### concatenate multifps videos ############################################
    """
    vis_dir = "/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_autoregressive_multifps/"

    datasets = sorted(os.listdir(vis_dir))

    for d in datasets:
        if ".mp4" in d:
            continue
        dvis_dir = os.path.join(vis_dir, d)
        sequences = sorted(os.listdir(dvis_dir))[::5]  # skipping 5 for 5 multifps videos
        vid_frames = []
        for i, seq in enumerate(sequences):
            framedir = os.path.join(dvis_dir, seq)
            frames = sorted(os.listdir(framedir))
            frames = frames[10:] + frames[:10]
            for f in frames:
                if "groundtruth" in f:
                    continue
                if "input_000" in f:
                    colored = 0
                    singlechannel = True
                else:
                    colored = 1
                    singlechannel = False
                framepath = os.path.join(framedir, f)
                frame = cv2.imread(framepath.replace('_10_depth_nogt', '_1_depth_nogt'), colored)
                frame1 = cv2.imread(framepath.replace('_10_depth_nogt', '_2_depth_nogt'), colored)
                frame2 = cv2.imread(framepath.replace('_10_depth_nogt', '_5_depth_nogt'), colored)
                frame3 = cv2.imread(framepath.replace('_10_depth_nogt', '_10_depth_nogt'), colored)
                frame4 = cv2.imread(framepath.replace('_10_depth_nogt', '_30_depth_nogt'), colored)
                h, w = frame.shape[:2]
                if singlechannel:
                    blank_image = np.zeros((h, w * 5)).astype(np.uint8)
                else:
                    blank_image = np.zeros((h, w * 5, 3)).astype(np.uint8)
                blank_image[:, 0 * w: 1 * w] = frame
                blank_image[:, 1 * w: 2 * w] = frame1
                blank_image[:, 2 * w: 3 * w] = frame2
                blank_image[:, 3 * w: 4 * w] = frame3
                blank_image[:, 4 * w: 5 * w] = frame4
                height, width = blank_image.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (0, 10)
                fontScale = 0.4
                color = (0, 0, 0)
                thickness = 1
                if singlechannel:
                    blank_image = np.dstack([blank_image, blank_image, blank_image])
                print(blank_image.shape)
                frame = cv2.putText(blank_image, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
                vid_frames.append(frame)

        fcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(os.path.join(vis_dir, d + "_collated.mp4"), fcc, 1.0, (width,height))
        for frame in vid_frames:
            video.write(frame)
        video.release()
    """


    vis_dir = "/data3/tkhurana/diffusers/logs/TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/checkpoint-9000/train_ddpm_autoregressive_multihyp_jan23_run1/"

    datasets = sorted(os.listdir(vis_dir))

    for d in datasets:
        if ".mp4" in d:
            continue
        dvis_dir = os.path.join(vis_dir, d)
        sequences = sorted(os.listdir(dvis_dir))
        vid_frames = []
        for i, seq in enumerate(sequences):
            if "_4_depth" in seq:
                continue
            print("Doing seq", seq)
            framedir = os.path.join(dvis_dir, seq)
            frames = sorted(os.listdir(framedir))[::-1]
            for f in frames:
                if "groundtruth" in f:
                    continue
                if "input_00001" in f or "input_00002" in f or "input_00003" in f:
                    continue
                framepath = os.path.join(framedir, f)
                frame = cv2.imread(framepath)
                frame1 = cv2.imread(framepath.replace('_run1', '_run2'))
                frame2 = cv2.imread(framepath.replace('_run1', '_run3'))
                frame3 = cv2.imread(framepath.replace('_run1', '_run4'))
                h, w = frame.shape[:2]
                blank_image = np.zeros((h, w * 4, 3)).astype(np.uint8)
                blank_image[:, 0 * w: 1 * w] = frame
                blank_image[:, 1 * w: 2 * w] = frame1
                blank_image[:, 2 * w: 3 * w] = frame2
                blank_image[:, 3 * w: 4 * w] = frame3
                height, width = blank_image.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (0, 10)
                fontScale = 0.4
                color = (0, 0, 0)
                thickness = 1
                frame = cv2.putText(blank_image, str(i), org, font, fontScale, color, thickness, cv2.LINE_AA)
                vid_frames.append(frame)

        fcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(os.path.join(vis_dir, d + "_2fps_collated.mp4"), fcc, 1.0, (width,height))
        for frame in vid_frames:
            video.write(frame)
        video.release()
