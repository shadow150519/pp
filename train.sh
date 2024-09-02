# TGN
# mkdir -p logs/TGN
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGN_CPU_600.yml > logs/TGN/tgn_wiki_2layer_600_10.log &
# nohup python -u train.py --gpu 1 --data REDDIT --config config/TGN_CPU_600.yml > logs/TGN/tgn_reddit_2layer_600_10.log &

batch_size="1000"
batch_size2="3000"
echo $batch_size
#python -u train.py --gpu 2 --data WIKI --config config/TGN_CPU_600.yml --bs ${batch_size} > logs/TGN/tgn_wiki_2layer_${batch_size}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/TGN_CPU_600.yml --bs ${batch_size} > logs/TGN/tgn_reddit_2layer_{$batch_size}_10_2.log
#wait
#python -u train.py --gpu 2 --data WIKI --config config/TGAT_CPU_600.yml --bs ${batch_size} > logs/TGAT/tgat_wiki_2layer_{$batch_size}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/TGAT_CPU_600.yml --bs ${batch_size} > logs/TGAT/tgat_reddit_2layer_{$batch_size}_10_2.log
#wait
#python -u train.py --gpu 2 --data WIKI --config config/APAN_CPU_600.yml --bs ${batch_size} > logs/APAN/apan_wiki_1layer_{$batch_size}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/APAN_CPU_600.yml --bs ${batch_size} > logs/APAN/apan_reddit_1layer_{$batch_size}_10_2.log
#wait

python -u train_partition.py --gpu 2 --data WIKI --config config/TGN_CPU_600.yml --bs ${batch_size} > logs/TGN/pdd_tgn_wiki_2layer_{$batch_size}_10_2.log
wait
python -u train_partition.py --gpu 2 --data REDDIT --config config/TGN_CPU_600.yml --bs ${batch_size} > logs/TGN/pdd_tgn_reddit_2layer_{$batch_size}_10_2.log
wait
python -u train_partition.py --gpu 2 --data WIKI --config config/TGAT_CPU_600.yml --bs ${batch_size} > logs/TGAT/pdd_tgat_wiki_2layer_{$batch_size}_10_2.log
wait
python -u train_partition.py --gpu 2 --data REDDIT --config config/TGAT_CPU_600.yml --bs ${batch_size} > logs/TGAT/pdd_tgat_reddit_2layer_{$batch_size}_10_2.log
wait
python -u train_partition.py --gpu 2 --data WIKI --config config/APAN_CPU_600.yml --bs ${batch_size} > logs/APAN/pdd_apan_wiki_1layer_{$batch_size}_10_2.log
wait
python -u train_partition.py --gpu 2 --data REDDIT --config config/APAN_CPU_600.yml --bs ${batch_size} > logs/APAN/pdd_apan_reddit_1layer_{$batch_size}_10_2.log




#python -u train.py --gpu 2 --data WIKI --config config/TGN_CPU_1800.yml --bs ${batch_size2} > logs/TGN/tgn_wiki_2layer_{$batch_size2}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/TGN_CPU_1800.yml --bs ${batch_size2} > logs/TGN/tgn_reddit_2layer_{$batch_size2}_10_2.log
#wait
#python -u train.py --gpu 2 --data WIKI --config config/TGAT_CPU_1800.yml --bs ${batch_size2} > logs/TGAT/tgat_wiki_2layer_{$batch_size2}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/TGAT_CPU_1800.yml --bs ${batch_size2} > logs/TGAT/tgat_reddit_2layer_{$batch_size2}_10_2.log
#wait
#python -u train.py --gpu 2 --data WIKI --config config/APAN_CPU_1800.yml --bs ${batch_size2} > logs/APAN/apan_wiki_1layer_{$batch_size2}_10_2.log
#wait
#python -u train.py --gpu 2 --data REDDIT --config config/APAN_CPU_1800.yml --bs ${batch_size2} > logs/APAN/apan_reddit_1layer_{$batch_size2}_10_2.log
#wait


# python -u train.py --gpu 2 --data GDELT --config config/APAN_CPU_600.yml > logs/APAN/apan_gdelt_1layer_600_10_1.log
# wait
# python -u train.py --gpu 3 --data GDELT --config config/TGAT_CPU_600.yml > logs/TGAT/tgat_gdelt_2layer_600_10_1.log 
# wait
# python -u train.py --gpu 2 --data GDELT --config config/TGN_CPU_600.yml > logs/TGN/tgn_gdelt_2layer_600_10_1.log 
# wait

# mkdir -p logs/TGN
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGN_CPU_600.yml > logs/TGN/tgn_wiki_2layer_600_10.log &
# nohup python -u train.py --gpu 1 --data REDDIT --config config/TGN_CPU_600.yml > logs/TGN/tgn_reddit_2layer_600_10.log &
# nohup python -u train.py --gpu 2 --data GDELT --config config/TGN_CPU_600.yml > logs/TGN/tgn_gdelt_2layer_600_10.log &

# # TGAT
# mkdir -p logs/TGAT
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGAT_CPU_600.yml > logs/TGAT/tgat_wiki_2layer_600_10.log &
# nohup python -u train.py --gpu 2 --data REDDIT --config config/TGAT_CPU_600.yml > logs/TGAT/tgat_reddit_2layer_600_10.log &
# nohup python -u train.py --gpu 3 --data GDELT --config config/TGAT_CPU_600.yml > logs/TGAT/tgat_gdelt_2layer_600_10.log &

# # APAN
# mkdir -p logs/APAN
# nohup python -u train.py --gpu 0 --data WIKI --config config/APAN_CPU_600.yml > logs/APAN/apan_wiki_1layer_600_10.log &
# nohup python -u train.py --gpu 1 --data REDDIT --config config/APAN_CPU_600.yml > logs/APAN/apan_reddit_1layer_600_10.log &
# nohup python -u train.py --gpu 2 --data GDELT --config config/APAN_CPU_600.yml > logs/APAN/apan_gdelt_1layer_600_10.log &





























# mkdir -p logs/TGN/wiki
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGN_CPU_200.yml > logs/TGN/wiki/tgn_wiki_2layer_200.log &
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGN_CPU_500.yml > logs/TGN/wiki/tgn_wiki_2layer_500.log &
# nohup python -u train.py --gpu 0 --data WIKI --config config/TGN_CPU_1000.yml > logs/TGN/wiki/tgn_wiki_2layer_1000.log &

# mkdir -p logs/TGN/reddit
# nohup python -u train.py --gpu 1 --data REDDIT --config config/TGN_CPU_200.yml > logs/TGN/reddit/tgn_reddit_2layer_200.log &
# nohup python -u train.py --gpu 2 --data REDDIT --config config/TGN_CPU_500.yml > logs/TGN/reddit/tgn_reddit_2layer_500.log &
# nohup python -u train.py --gpu 3 --data REDDIT --config config/TGN_CPU_1000.yml > logs/TGN/reddit/tgn_reddit_2layer_1000.log &

# mkdir -p logs/TGN/gdelt
# nohup python -u train.py --gpu 0 --data GDELT --config config/TGN_CPU_200_gdelt.yml > logs/TGN/gdelt/tgn_gdelt_1layer_200.log &
# nohup python -u train.py --gpu 1 --data GDELT --config config/TGN_CPU_500_gdelt.yml > logs/TGN/gdelt/tgn_gdelt_1layer_500.log &
# nohup python -u train.py --gpu 2 --data GDELT --config config/TGN_CPU_1000_gdelt.yml > logs/TGN/gdelt/tgn_gdelt_1layer_1000.log &

# nohup python -u train.py --gpu 3 --data GDELT --config config/TGN_CPU_200.yml > logs/TGN/gdelt/tgn_gdelt_2layer_200.log &
# nohup python -u train.py --gpu 1 --data GDELT --config config/TGN_CPU_500.yml > logs/TGN/gdelt/tgn_gdelt_2layer_500.log &
# nohup python -u train.py --gpu 0 --data GDELT --config config/TGN_CPU_1000.yml > logs/TGN/gdelt/tgn_gdelt_2layer_1000.log &

# # TGAT
# mkdir -p logs/TGAT/wiki
# nohup python -u train.py --gpu 1 --data WIKI --config config/TGAT_CPU_200.yml > logs/TGAT/wiki/tgat_wiki_2layer_200.log &
# nohup python -u train.py --gpu 2 --data WIKI --config config/TGAT_CPU_500.yml > logs/TGAT/wiki/tgat_wiki_2layer_500.log &
# nohup python -u train.py --gpu 3 --data WIKI --config config/TGAT_CPU_1000.yml > logs/TGAT/wiki/tgat_wiki_2layer_1000.log &

# mkdir -p logs/TGAT/reddit
# nohup python -u train.py --gpu 1 --data REDDIT --config config/TGAT_CPU_200.yml > logs/TGAT/reddit/tgat_reddit_2layer_200.log &
# nohup python -u train.py --gpu 2 --data REDDIT --config config/TGAT_CPU_500.yml > logs/TGAT/reddit/tgat_reddit_2layer_500.log &
# nohup python -u train.py --gpu 3 --data REDDIT --config config/TGAT_CPU_1000.yml > logs/TGAT/reddit/tgat_reddit_2layer_1000.log &

# mkdir -p logs/TGAT/gdelt
# nohup python -u train.py --gpu 0 --data GDELT --config config/TGAT_CPU_200_gdelt.yml > logs/TGAT/gdelt/tgat_gdelt_1layer_200.log &
# nohup python -u train.py --gpu 1 --data GDELT --config config/TGAT_CPU_500_gdelt.yml > logs/TGAT/gdelt/tgat_gdelt_1layer_500.log &
# nohup python -u train.py --gpu 2 --data GDELT --config config/TGAT_CPU_1000_gdelt.yml > logs/TGAT/gdelt/tgat_gdelt_1layer_1000.log &

# nohup python -u train.py --gpu 3 --data GDELT --config config/TGAT_CPU_200.yml > logs/TGAT/gdelt/tgat_gdelt_2layer_200.log &
# nohup python -u train.py --gpu 1 --data GDELT --config config/TGAT_CPU_500.yml > logs/TGAT/gdelt/tgat_gdelt_2layer_500.log &
# nohup python -u train.py --gpu 0 --data GDELT --config config/TGAT_CPU_1000.yml > logs/TGAT/gdelt/tgat_gdelt_2layer_1000.log &



