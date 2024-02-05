BASE_PATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project
MP=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/hypothesis_1/output/models/Xception_three_class/Xception_three_class.h5

# multiline run data formatter
python3 utils/data_formatter.py \
    -i /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/rgb \
    -m /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/master/NuCLSEvalSet/csv \
    -s /media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/yolov5_three_class \
    -p run \
    -mp $MP \
    -v three_class \
    -f 3 \
    --seed 42 \