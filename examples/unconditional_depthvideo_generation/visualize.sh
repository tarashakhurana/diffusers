# python visualize.py \
#     --model_dir /data3/tkhurana/diffusers/logs/PointOdyssey-depth_minitrain_6s_random-half_masking_resolution64_stdconditioningwithcrossattn_debugwithframenumbers/ \
#     --checkpoint_number 5000 \
#     --model_type depthpose \
#     --eval_data_dir /data/tkhurana/datasets/pointodyssey/minitrain/ \
#     --train_with_plucker_coords \
#     --use_rendering \
#     --masking_strategy random-half \
#     --out_channels 12 \
#     --in_channels 36

python visualize.py \
    --model_dir /data3/tkhurana/diffusers/logs/TAO-rgbd_val_img2img_resolution-64_stdunetwithcrossattn_futuresinglestepprediction_sequentialcontext_2shorizon_startfromSD_finetuneverything/ \
    --checkpoint_number 5000 \
    --model_type img2img \
    --eval_data_dir /data/tkhurana/TAO-depth/zoe/frames/val/ \
    --eval_rgb_data_dir /data3/chengyeh/TAO/frames/val/ \
    --data_format rgbd \
    --train_with_plucker_coords \
    --use_rendering \
    --in_channels 16 \
    --out_channels 4 \
    --n_input 3 \
    --n_output 1 \
    --num_images 3 \
    --normalization_factor 20480.0 \
    --masking_strategy custom
