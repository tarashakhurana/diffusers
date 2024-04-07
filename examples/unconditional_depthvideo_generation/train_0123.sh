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
    --train_data_dir /data/tkhurana/TAO-depth/zoe/frames/val/ \
    --masking_strategy custom \
    --train_batch_size 12 \
    --output_dir /data3/tkhurana/diffusers/logs/TAO-rgb-grayscale_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/ \
    --resolution 64 \
    --checkpointing_steps 1000 \
    --p_unconditional 0.1 \
    --loss_in_2d \
    --in_channels 4 \
    --out_channels 1 \
    --num_images 3 \
    --n_input 3 \
    --n_output 1 \
    --data_format rgb \
    --train_rgb_data_dir /data3/chengyeh/TAO/frames/val/ \
    --train_with_plucker_coords \
    --co3d_annotations_root /compute/trinity-2-5/tkhurana/co3d_v2_recopy/ \
    --co3d_rgb_data_root /compute/trinity-2-25/tkhurana/datasets/TAO/frames/valco3d/ \
    --use_rendering



# accelerate launch --main_process_port 29500 train_0123.py \
#     --train_data_dir /data/tkhurana/TAO-depth/zoe/frames/val/ \
#     --masking_strategy custom \
#     --train_batch_size 16 \
#     --output_dir /data3/tkhurana/diffusers/logs/TAO-depth_val_10sfuture_regressionbaseline_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromSD_finetuneverything/ \
#     --resolution 64 \
#     --checkpointing_steps 1000 \
#     --p_unconditional 0.1 \
#     --loss_in_2d \
#     --in_channels 4 \
#     --out_channels 1 \
#     --num_images 3 \
#     --n_input 3 \
#     --n_output 1 \
#     --data_format d \
#     --prediction_type sample \
#     --train_rgb_data_dir /data3/chengyeh/TAO/frames/val/ \
#     --train_with_plucker_coords \
#     --co3d_annotations_root /compute/trinity-2-5/tkhurana/co3d_v2_recopy/ \
#     --co3d_rgb_data_root /compute/trinity-2-25/tkhurana/datasets/TAO/frames/valco3d/ \
#     --use_rendering \
#     --mixed_precision fp16





