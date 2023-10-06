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


accelerate launch --main_process_port 29502 train_inpainting.py \
    --train_data_dir /data/tkhurana/TAO-depth/zoe/frames/minitrain/ \
    --masking_strategy random \
    --train_batch_size 8 \
    --output_dir /data3/tkhurana/diffusers/logs/TAO-depth_minitrain_custom-masking_resolution-64_stdunetwithcrossattn_maskedMAE_startfromSD_finetuneverything/ \
    --resolution 64 \
    --checkpointing_steps 1000 \
    --loss_in_2d \
    --in_channels 36 \
    --out_channels 12 \
    --train_with_plucker_coords \
    --use_rendering


