#!/bin/bash

# 模型列表
models=("TGN" "APAN" "TGAT")

# 数据集列表
datasets=("WIKI" "REDDIT" "MOOC" "GDELT")

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    # 遍历每个模型
    for model in "${models[@]}"; do
        # 如果数据集是 GDELT，跳过 GPU 配置
        if [ "$dataset" == "GDELT" ]; then
            configs=("${model}_CPU.yml")
        else
            configs=("${model}_GPU.yml" "${model}_CPU.yml")
        fi
        # 遍历每个配置文件
        for config in "${configs[@]}"; do
            echo "Start: $dataset with $config"
            config_name=$(basename $config .yml)
            log_dir="cache_logs/memprofile_3000_2/$dataset"
            log_file="$log_dir/${dataset}_${config_name}_mem_profile.log"
            
            # 创建日志目录（如果不存在）
            mkdir -p $log_dir
            
            # 运行训练命令
            python -u train.py --gpu 0 --data $dataset --config config/$config > $log_file
            wait          
            
        done
    done
done
