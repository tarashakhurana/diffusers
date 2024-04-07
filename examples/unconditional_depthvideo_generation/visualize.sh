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
    --model_dir /data3/tkhurana/diffusers/logs//TAO-depth_val_0123_bugfix_resolution-64_stdunetwithcrossattn_singlestepprediction_randomsequence_4shorizon_startfromIV_finetuneverything_10xlrforscratchlayers_puncond0.1/ \
    --checkpoint_number 9000 \
    --model_type clippose \
    --eval_data_dir /compute/trinity-2-25/tkhurana/datasets/data/data/tkhurana/TAO-depth/zoe/frames/train/ \
    --eval_rgb_data_dir /data3/chengyeh/TAO/frames/train/ \
    --data_format d \
    --train_with_plucker_coords \
    --use_rendering \
    --in_channels 4 \
    --out_channels 1 \
    --n_input 3 \
    --n_output 1 \
    --num_images 3 \
    --prediction_type epsilon \
    --num_inference_steps 40 \
    --normalization_factor 20480.0 \
    --masking_strategy custom \
    --co3d_annotations_root /ssd0/jasonzh2/co3d_v2_recopy/ \
    --co3d_rgb_data_root /compute/trinity-2-25/tkhurana/datasets/TAO/frames/trainco3d/ \
    --guidance 2.0
