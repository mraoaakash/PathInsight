# INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
# python3 utils/json_gen.py \
#     --path $INDIR 


INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/faster_rcnn/csvs
python3 utils/generate_csv.py \
    -d $INDIR \
    -o $OUTDIR \



INDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/faster_rcnn/csvs
OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/faster_rcnn/plots
python3 utils/plot_faster_rcnn.py \
    -d $INDIR \
    -o $OUTDIR

# OLDDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/capstone_run_archive/detectron/plots/faster_rcnn/csvs
# NEWDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/faster_rcnn/csvs
# OUTDIR=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/plots/faster_rcnn/comparative
# python3 utils/comparative.py \
#     -d $OLDDIR \
#     -o $OUTDIR \
#     -n $NEWDIR 