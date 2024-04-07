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

# for ITERATION in 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
# do
python evaluate_hierarchical.py \
        --model_dir /data3/tkhurana/diffusers/logs/TAO-rgb-grayscale_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/ \
        --checkpoint_number 9000 \
        --model_type clippose \
        --eval_data_dir /data/tkhurana/TAO-depth/zoe/frames/train/ \
        --eval_rgb_data_dir /data3/chengyeh/TAO/frames/train/ \
        --data_format rgb \
        --train_with_plucker_coords \
        --use_rendering \
        --in_channels 4 \
        --out_channels 1 \
        --n_input 3 \
        --n_output 1 \
        --num_images 3 \
        --normalization_factor 20480.0 \
        --masking_strategy custom \
        --guidance 2.0 \
        --sampling_strategy hierarchical \
        --num_autoregressive_frames 10 \
        --num_inference_steps 40 \
        --output_fps 1
# done
