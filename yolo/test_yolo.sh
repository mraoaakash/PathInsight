# NuCLSEvalSet
# read a text file 
input='./api_keys.txt'
export COMET_API_KEY=$(cat "$input")
export COMET_WORKSPACE=mraoaakash
export COMET_PROJECT_NAME=capstone-project-final
# setting variables
YAML_FOLD_1=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_1.yaml
YAML_FOLD_2=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_2.yaml
YAML_FOLD_3=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_3.yaml

IMG_SIZE=520
EPOCHS=200
BATCH_SIZE=16
DEVICE=0
SAVE_PERIOD=10
PROJECT=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo_test
INPATH=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo


echo "Training YOLOv5s"
echo $input
echo $COMET_API_KEY
echo $YAML_FOLD_1
echo $YAML_FOLD_2
echo $YAML_FOLD_3
echo $IMG_SIZE
echo $EPOCHS
echo $BATCH_SIZE
echo $DEVICE
echo $SAVE_PERIOD

# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_1 --weights $INPATH/yolov5m-fold_1/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5m-fold_1_test --task val --save-txt --save-conf --save-hybrid --save-json --conf-thres 0.5 --iou-thres 0.5
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_2 --weights $INPATH/yolov5m-fold_2/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5m-fold_2_test --task test --save-txt --save-conf --save-hybrid --save-json
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_3 --weights $INPATH/yolov5m-fold_3/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5m-fold_3_test --task test --save-txt --save-conf --save-hybrid --save-json


python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_1 --weights $INPATH/yolov5l-fold_1/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5l-fold_1_test --task test --save-txt --save-conf --save-hybrid --save-json --conf-thres 0.5 --iou-thres 0.5 --verbose
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_2 --weights $INPATH/yolov5l-fold_2/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5l-fold_2_test --task test --save-txt --save-conf --save-hybrid --save-json
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_3 --weights $INPATH/yolov5l-fold_3/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5l-fold_3_test --task test --save-txt --save-conf --save-hybrid --save-json


python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_1 --weights $INPATH/yolov5x-fold_1/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5x-fold_1_test --task test --save-txt --save-conf --save-hybrid --save-json --conf-thres 0.5 --iou-thres 0.5 --verbose
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_2 --weights $INPATH/yolov5x-fold_2/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5x-fold_2_test --task test --save-txt --save-conf --save-hybrid --save-json
# python ./yolov5/val.py --img $IMG_SIZE --data $YAML_FOLD_3 --weights $INPATH/yolov5x-fold_3/weights/best.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --name yolov5x-fold_3_test --task test --save-txt --save-conf --save-hybrid --save-json
