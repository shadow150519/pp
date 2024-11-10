#!/bin/bash

# for dataset in "REDDIT" "MOOC" "LASTFM" "WIKI"
# for dataset in "GDELT"
for dataset in "WIKI" "REDDIT" "MOOC" "LASTFM"
do
    for cfg in "JODIE" "APAN" "TGAT" "TGN"
    do
        if [ "${cfg}" = "TGAT" ] && { [ "${dataset}" = "LASTFM" ] || [ "${dataset}" = "MOOC" ]; }; then
            continue
        fi
        for bs in 1000 3000 5000
        do
            cfg_path="config/${cfg}.yml"
            for drop_rate in 0.1 0.25 0.5
            do
              for i in {3..4}
              do
                log_path="partition_log_drop_random/${cfg}"
                log_file="partition_log_drop_random/${cfg}/${dataset}_${bs}_${drop_rate}_${i}.log"
                mkdir -p ${log_path}
                gpu=$((i%2))
                echo "${cfg} ${dataset} ${bs} ${drop_rate} ${i} ${gpu} start at `date +"%Y-%m-%d %H:%M:%S"`"
                python -u train_drop_random.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path} --tolerance 5 --nepoch 100 --drop_rate ${drop_rate}> $log_file &
              done
              wait
            done
        done
    done
done