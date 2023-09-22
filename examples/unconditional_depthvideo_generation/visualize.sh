python visualize.py \
    --model_dir /data3/tkhurana/diffusers/logs/PointOdyssey-depth_minitrain_6s_random_masking_resolution64_with_correct_plucker_poseatrendering_renderoutchannels12_attentionwithMLP/ \
    --checkpoint_number 2500 \
    --model_type depthpose \
    --eval_data_dir /data/tkhurana/datasets/pointodyssey/minitrain/ \
    --train_with_plucker_coords \
    --use_rendering \
    --prediction_type sample \
    --out_channels 72 \
    --in_channels 36
