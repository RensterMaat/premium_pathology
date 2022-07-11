#!/bin/bash

cd "/hpc/dla_patho/premium/wliu/hover_net"
# cd "/hpc/dla_patho/premium/PREMIUM histopathology/data/LUMC/test"

python run_infer.py \
    --gpu='0,1' \
    --model_path="weights/hovernet_fast_pannuke_type_tf2pytorch.tar" \
    --nr_types=6 \
    --batch_size=32 \
    --type_info_path=type_info.json \
    --nr_inference_workers=4 \
    --nr_post_proc_workers=8 \
    --model_mode='fast' \
    wsi \
    --input_dir="/hpc/dla_patho/premium/tcga_skcm/test" \
    --cache_path='/hpc/dla_patho/premium/rens/output/tmp' \
    --output_dir="/hpc/dla_patho/premium/tcga_skcm/output" \

    
