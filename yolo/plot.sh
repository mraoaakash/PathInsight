INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/yolo/csvs
python3 utils/generate_csvs.py \
    -d $INDIR \
    -o $OUTDIR


INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/yolo/csvs
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/yolo/plots
python3 utils/plot_yolo.py \
    -d $INDIR \
    -o $OUTDIR


# OLDDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_run_archive/detectron/plots/yolo/csvs
# NEWDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/yolo/csvs
# OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/yolo/comparative
# python3 utils/comparative.py \
#     -d $OLDDIR \
#     -o $OUTDIR \
#     -n $NEWDIR 