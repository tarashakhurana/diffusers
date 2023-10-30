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


accelerate launch --main_process_port 29500 train_0123.py \
    --train_data_dir /scratch/tkhurana/datasets/data/data/tkhurana/TAO-depth/zoe/frames/val/ \
    --masking_strategy custom \
    --train_batch_size 32 \
    --output_dir /data3/tkhurana/diffusers/logs/TAO-rgbd-grayscale_val_img2img_resolution-64_smallerunetwithcrossattn_singlestepprediction_randomsequence_2shorizon/ \
    --resolution 64 \
    --checkpointing_steps 1000 \
    --p_unconditional 0.1 \
    --resume_from_checkpoint latest \
    --loss_in_2d \
    --in_channels 8 \
    --out_channels 2 \
    --num_images 3 \
    --n_input 3 \
    --n_output 1 \
    --data_format rgbd \
    --train_rgb_data_dir /data3/chengyeh/TAO/frames/val/ \
    --train_with_plucker_coords \
    --use_rendering


