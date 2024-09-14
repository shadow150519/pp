#!/bin/bash

declare -A bs_dict

bs_dict["REDDIT_500"]=1322
bs_dict["MOOC_500"]=511
bs_dict["LASTFM_500"]=1100
bs_dict["WIKI_500"]=1069
bs_dict["REDDIT_1000"]=2645
bs_dict["MOOC_1000"]=1020
bs_dict["LASTFM_1000"]=2200
bs_dict["WIKI_1000"]=2120
bs_dict["REDDIT_2000"]=5262
bs_dict["MOOC_2000"]=2040
bs_dict["LASTFM_2000"]=4400
bs_dict["WIKI_2000"]=4241
bs_dict["REDDIT_3000"]=7937
bs_dict["MOOC_3000"]=3036
bs_dict["LASTFM_3000"]=6600
bs_dict["WIKI_3000"]=6362


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
                log_path="partition_log2/${cfg}"
                log_file="partition_log2/${cfg}/${dataset}_${bs}_${i}.log"
                log_file2="partition_log2/${cfg}/${dataset}_${bs_dict["${dataset}_${bs}"]}_${i}.log"
                mkdir -p ${log_path}
                gpu=$((i%2))
                echo "${cfg} ${dataset} ${bs} ${i} ${gpu}"
                echo "${cfg} ${dataset} ${bs_dict["${dataset}_${bs}"]} ${i} ${gpu}"
                python -u train.py --gpu ${gpu} --data $dataset --bs ${bs} --config ${cfg_path}  > $log_file &
                python -u train.py --gpu ${gpu} --data $dataset --bs ${bs_dict["${dataset}_${bs}"]} --config ${cfg_path} > $log_file2 &
            done
            wait
        done
    done
done