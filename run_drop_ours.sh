#!/bin/bash

# for dataset in "WIKI" "REDDIT" "MOOC" "LASTFM"
for dataset in "REDDIT" "MOOC" "LASTFM" "WIKI"
do
    for cfg in "JODIE" "APAN" "TGAT" "TGN"
    do
        if [ "${cfg}" = "TGAT" ] && { [ "${dataset}" = "LASTFM" ] || [ "${dataset}" = "MOOC" ]; }; then
            continue
        fi
        for bs in 500 1000 2000 3000 4000 5000
        do
            cfg_path="config/${cfg}.yml"
            for i in {1..2}
            do
                log_path="partition_log_drop_ours/${cfg}"
                log_file="partition_log_drop_ours/${cfg}/${dataset}_${bs}_${i}.log"
                mkdir -p ${log_path}
                gpu=$((i%2))
                echo "${cfg} ${dataset} ${bs} ${i} ${gpu}"
                if [ "${dataset}" = "WIKI" ] || [ "${dataset}" = "REDDIT" ];then
                  python -u train_drop_ours.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path} --pnum 10 --nepoch 150 --partition_interval 10 > $log_file &
                else
                  python -u train_drop_ours.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path} --pnum 5  --nepoch 150 --partition_interval 10 > $log_file &
                fi
            done
            wait
        done
    done
done