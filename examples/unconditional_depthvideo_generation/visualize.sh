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
    --model_dir /data3/tkhurana/diffusers/logs/TAO-depth_minitrain_custom-masking_resolution-64_stdunetwithcrossattn_predictonlyonearbitraryfuture/ \
    --checkpoint_number 7000 \
    --model_type depthpose \
    --eval_data_dir /data/tkhurana/TAO-depth/zoe/frames/minitrain/ \
    --train_with_plucker_coords \
    --use_rendering \
    --in_channels 12 \
    --out_channels 4 \
    --n_input 3 \
    --n_output 1 \
    --num_images 3 \
    --normalization_factor 20480.0 \
    --masking_strategy random
