#!/bin/bash

# for dataset in "WIKI" "REDDIT" "MOOC" "LASTFM"
for dataset in "REDDIT" "MOOC" "LASTFM" "WIKI"
do
    for cfg in "JODIE" "APAN" "TGAT" "TGN"
    do
        for bs in 500 1000 2000 3000
        do
            cfg_path="config/${cfg}.yml"
            for i in {1..2}
            do
                log_path="partition_log/${cfg}"
                log_file="partition_log/${cfg}/${dataset}_${bs}_${i}.log"
                mkdir -p ${log_path}
                gpu=$((i%2))
                echo "${cfg} ${dataset} ${bs} ${i} ${gpu}"
                if [ "${dataset}" = "WIKI" ] || [ "${dataset}" = "REDDIT" ];then
                  python -u train_partition.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path} --pnum 10 --maxpnum 3 --tolerance 0.4 --schedule_alg pp2  > $log_file &
                else
                  python -u train_partition.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path} --pnum 5 --maxpnum 2 --tolerance 0.4 --schedule_alg pp2  > $log_file &
                fi
            done
            wait
        done
    done
done