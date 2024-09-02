# REDDIT 11k 672K
 nohup python -u train.py --gpu 0 --data REDDIT --config config/TGN.yml > cache_logs/reddit/REDDIT_cpu.log &
 nohup python train_cache.py --gpu 0 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "LRU" --disn > cache_logs/reddit/cache_REDDIT_16_LRU_edge.log & # 10%
 nohup python train_cache.py --gpu 0 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "LRU" --disn > cache_logs/reddit/cache_REDDIT_17_LRU_edge.log & # 19.5%
 nohup python train_cache.py --gpu 0 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 18 --ecstrategy "LRU" --disn > cache_logs/reddit/cache_REDDIT_18_LRU_edge.log & # 39%
 nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 19 --ecstrategy "LRU" --disn > cache_logs/reddit/cache_REDDIT_19_LRU_edge.log & # 7cache
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "LRU"  > cache_logs/reddit/cache_REDDIT_16_LRU_edge_and_node.log & # 10%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "LRU"  > cache_logs/reddit/cache_REDDIT_17_LRU_edge_and_node.log & # 19.5%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 18 --ecstrategy "LRU"  > cache_logs/reddit/cache_REDDIT_18_LRU_edge_and_node.log & # 39%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 19 --ecstrategy "LRU"  > cache_logs/reddit/cache_REDDIT_19_LRU_edge_and_node.log & # 7cache

# # # REDDIT 11k 672K FIFO
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 16 --ecstrategy "FIFO" --disn > cache_logs/cache_REDDIT_16_FIFO_edge.log & # 10%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 17 --ecstrategy "FIFO" --disn > cache_logs/cache_REDDIT_17_FIFO_edge.log & # 19.5%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 18 --ecstrategy "FIFO" --disn > cache_logs/cache_REDDIT_18_FIFO_edge.log & # 39%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 19 --ecstrategy "FIFO" --disn > cache_logs/cache_REDDIT_19_FIFO_edge.log & # 78

# # # R & #EDDIT 11k 672K LFU
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 16 --ecstrategy "LFU" --disn > cache_logs/cache_REDDIT_16_LFU_edge.log & # 10%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 17 --ecstrategy "LFU" --disn > cache_logs/cache_REDDIT_17_LFU_edge.log & # 19.5%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 18 --ecstrategy "LFU" --disn > cache_logs/cache_REDDIT_18_LFU_edge.log & # 39%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 19 --ecstrategy "LFU" --disn > cache_logs/cache_REDDIT_19_LFU_edge.log & # 78


# # WIKI 9k 157K
# nohup python train_cache.py --gpu 0 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 14 --ecstrategy "LRU" > cache_logs/new_WIKI_14_LRU.log & # 10.43%
# nohup python train_cache.py --gpu 0 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 15 --ecstrategy "LRU" > cache_logs/new_WIKI_15_LRU.log & # 20.87%
# nohup python train_cache.py --gpu 0 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "LRU" > cache_logs/new_WIKI_16_LRU.log & # 41.74%
# nohup python train_cache.py --gpu 0 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "LRU" > cache_logs/new_WIKI_17_LRU.log & # 83.48%
#
#
# nohup python train_cache.py --gpu 1 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 14 --ecstrategy "FIFO" > cache_logs/new_WIKI_14_FIFO.log & # 10.43%
# nohup python train_cache.py --gpu 1 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 15 --ecstrategy "FIFO" > cache_logs/new_WIKI_15_FIFO.log & # 20.87%
# nohup python train_cache.py --gpu 1 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "FIFO" > cache_logs/new_WIKI_16_FIFO.log & # 41.74%
# nohup python train_cache.py --gpu 1 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "FIFO" > cache_logs/new_WIKI_17_FIFO.log & # 83.48%
#
# nohup python train_cache.py --gpu 2 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 14 --ecstrategy "LFU" > cache_logs/new_WIKI_14_LFU.log & # 10.43%
# nohup python train_cache.py --gpu 2 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 15 --ecstrategy "LFU" > cache_logs/new_WIKI_15_LFU.log & # 20.87%
# nohup python train_cache.py --gpu 2 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "LFU" > cache_logs/new_WIKI_16_LFU.log & # 41.74%
# nohup python train_cache.py --gpu 2 --data WIKI --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "LFU" > cache_logs/new_WIKI_17_LFU.log & # 83.48%
#
#
#

# GDELT 17k 191M
#  nohup python -u train_cache.py --gpu 0 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 22 --ecstrategy "LRU" --disn > cache_logs/GDELT_22_LRU_edge.log & # 3.46%
#  nohup python -u train_cache.py --gpu 1 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 23 --ecstrategy "LRU" --disn > cache_logs/GDELT_23_LRU_edge.log & # 6.93%
#  nohup python -u train_cache.py --gpu 2 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 24 --ecstrategy "LRU" --disn > cache_logs/GDELT_24_LRU_edge.log & # 13.86%
#  nohup python -u train_cache.py --gpu 3 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 25 --ecstrategy "LRU" --disn > cache_logs/GDELT_25_LRU_edge.log & # 27.72%

# python train_cache.py --gpu 0 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 23 --ecstrategy "LRU" --disn  # 8.78%
# python train_cache.py --gpu 1 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 25 --ecstrategy "LRU" --disn  # 17.56%
# python train_cache.py --gpu 2 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 26 --ecstrategy "LRU" --disn  # 35.12%
# python -u train.py --gpu 1 --data GDELT --config config/TGN.yml