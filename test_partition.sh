#nohup python partition.py --data "WIKI" > partition_logs/wiki.log &
#nohup python partition.py --data "REDDIT" > partition_logs/reddit.log &
#nohup python partition.py --data "LASTFM" > partition_logs/lastfm.log &
#nohup python partition.py --data "MOOC" > partition_logs/mooc.log &
# nohup python partition.py --data "GDELT" > partition_logs/gdelt.log &
# nohup python partition.py --data "MAG" > partition_logs/mag.log &

# product
#nohup python partition.py --data "amazon0302" > partition_logs/amazon0302.txt.log &
#nohup python partition.py --data "amazon0312" > partition_logs/amazon0312.txt.log &
#nohup python partition.py --data "amazon0505" > partition_logs/amazon0505.txt.log &
#nohup python partition.py --data "amazon0601" > partition_logs/amazon0601.txt.log &

# community
#nohup python partition.py --data "orkut" > partition_logs/orkut.log &
#nohup python partition.py --data "facebook" > partition_logs/facebook.log &
#nohup python partition.py --data "gplus" > partition_logs/gplus.log &
#nohup python partition.py --data "twitter" > partition_logs/twitter.log &
#nohup python partition.py --data "livejournal" > partition_logs/livejournal.log &
#nohup python partition.py --data "pokec" > partition_logs/pokec.log &

# web
nohup python partition.py --data "Google" > partition_logs/google.log &
nohup python partition.py --data "BerkStan" > partition_logs/berkstan.log &
nohup python partition.py --data "Stanford" > partition_logs/stanford.log &

# cite
nohup python partition.py --data "HepPh" > partition_logs/hepph.log &
nohup python partition.py --data "Patents" > partition_logs/patents.log &