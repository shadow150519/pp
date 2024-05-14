# REDDIT 11k 672K
 nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 16 --ecstrategy "LRU" --disn > cache_logs/new_REDDIT_16_LRU.log & # 10%
 nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 17 --ecstrategy "LRU" --disn > cache_logs/new_REDDIT_17_LRU.log & # 19.5%
 nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 18 --ecstrategy "LRU" --disn > cache_logs/new_REDDIT_18_LRU.log & # 39%
 nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 19 --ecstrategy "LRU" --disn > cache_logs/new_REDDIT_19_LRU.log & # 78%


# # # REDDIT 11k 672K FIFO
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 16 --ecstrategy "FIFO" > cache_logs/new_REDDIT_16_FIFO.log & # 10%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 17 --ecstrategy "FIFO" > cache_logs/new_REDDIT_17_FIFO.log & # 19.5%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 18 --ecstrategy "FIFO" > cache_logs/new_REDDIT_18_FIFO.log & # 39%
#  nohup python train_cache.py --gpu 1 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 19 --ecstrategy "FIFO" > cache_logs/new_REDDIT_19_FIFO.log & # 78


# # R & #EDDIT 11k 672K LFU
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 16 --ecstrategy "LFU" > cache_logs/new_REDDIT_16_LFU.log & # 10%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 17 --ecstrategy "LFU" > cache_logs/new_REDDIT_17_LFU.log & # 19.5%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 18 --ecstrategy "LFU" > cache_logs/new_REDDIT_18_LFU.log & # 39%
#  nohup python train_cache.py --gpu 2 --data REDDIT --config config/TGN.yml --ncsize 4096 --ncstrategy "FIFO" --ecsize 19 --ecstrategy "LFU" > cache_logs/new_REDDIT_19_LFU.log & # 78


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
#  nohup python train_cache.py --gpu 0 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 23 --ecstrategy "LRU" --disn > cache_logs/GDELT_23_LRU.log & # 8.78%
#  nohup python train_cache.py --gpu 1 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 25 --ecstrategy "LRU" --disn > cache_logs/GDELT_25_LFU.log & # 17.56%
#  nohup python train_cache.py --gpu 2 --data GDELT --config config/TGN.yml --ncsize 4096 --ncstrategy "LRU" --ecsize 26 --ecstrategy "LRU" --disn > cache_logs/GDELT_26_LFU.log & # 35.12%