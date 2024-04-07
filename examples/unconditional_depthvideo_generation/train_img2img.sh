# accelerate launch --main_process_port 29502 train_inpainting.py \
#     --train_data_dir /data/tkhurana/datasets/pointodyssey/minitrain/ \
#     --masking_strategy random-half \
#     --train_batch_size 2 \
#     --output_dir /data3/tkhurana/diffusers/logs/PointOdyssey-depth_minitrain_6s_random-half_masking_resolution64_stdconditioningwithcrossattn_debugwithframenumbers/ \
#     --resolution 64 \
#     --checkpointing_steps 1000 \
#     --normalization_factor 65.535 \
#     --loss_in_2d \
#     --in_channels 36 \
#     --out_channels 12 \
#     --learning_rate 1e-4 \
#     --train_with_plucker_coords \
#     --use_rendering


accelerate launch --main_process_port 29502 train_img2img.py \
    --train_data_dir /scratch/tkhurana/datasets/data/data/tkhurana/TAO-depth/zoe/frames/val/ \
    --masking_strategy custom \
    --train_batch_size 12 \
    --output_dir /data3/tkhurana/diffusers/logs/TAO-rgb-grayscale_val_regression_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_2shorizon_startfromSD_finetuneverything/ \
    --resolution 64 \
    --checkpointing_steps 1000 \
    --loss_in_2d \
    --in_channels 4 \
    --out_channels 1 \
    --num_images 3 \
    --n_input 3 \
    --n_output 1 \
    --data_format rgb \
    --prediction_type sample \
    --normalization_factor 20480.0 \
    --train_rgb_data_dir /data3/chengyeh/TAO/frames/val/ \
    --train_with_plucker_coords \
    --use_rendering
