eval "$(conda shell.bash hook)"
conda activate yolov5
# NuCLSEvalSet
# read a text file 
input='./api_keys.txt'
export COMET_API_KEY=$(cat "$input")
export COMET_WORKSPACE=mraoaakash
export COMET_PROJECT_NAME=capstone-project-final
# setting variables
# YAML_FOLD_1=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_1.yaml
# YAML_FOLD_2=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_2.yaml
# YAML_FOLD_3=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_3.yaml

# IMG_SIZE=520
# EPOCHS=150
# BATCH_SIZE=16
# DEVICE=0
# SAVE_PERIOD=10
# PROJECT=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo

# echo "Training YOLOv5s"
# echo $input
# echo $COMET_API_KEY
# echo $YAML_FOLD_1
# echo $YAML_FOLD_2
# echo $YAML_FOLD_3
# echo $IMG_SIZE
# echo $EPOCHS
# echo $BATCH_SIZE
# echo $DEVICE
# echo $SAVE_PERIOD

# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_1
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_2
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_3

# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights yolov5l.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5l-fold_1
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights yolov5l.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5l-fold_2
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights yolov5l.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5l-fold_3

# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights yolov5x.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5x-fold_1
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights yolov5x.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5x-fold_2
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights yolov5x.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5x-fold_3






# YAML_FOLD_1=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_1_single.yaml
# YAML_FOLD_2=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_2_single.yaml
# YAML_FOLD_3=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_3_single.yaml

# IMG_SIZE=520
# EPOCHS=150
# BATCH_SIZE=16
# DEVICE=0
# SAVE_PERIOD=10
# PROJECT=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_1-single
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_2-single
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_3-single


# YAML_FOLD_1=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_1_three_class.yaml
# YAML_FOLD_2=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_2_three_class.yaml
# YAML_FOLD_3=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_3_three_class.yaml

# IMG_SIZE=520
# EPOCHS=150
# BATCH_SIZE=16
# DEVICE=0
# SAVE_PERIOD=10
# PROJECT=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_1-three_class
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_2-three_class
# python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights yolov5m.pt --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_3-three_class



YAML_FOLD_1=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_1_three_class.yaml
YAML_FOLD_2=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_2_three_class.yaml
YAML_FOLD_3=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/yolo/configs/fold_3_three_class.yaml

IMG_SIZE=520
EPOCHS=150
BATCH_SIZE=16
DEVICE=0
SAVE_PERIOD=10
PROJECT=/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/yolo
python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_2 --weights '' --cfg yolov5m.yaml --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_2-three_class_random_weights
python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_3 --weights '' --cfg yolov5m.yaml --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_3-three_class_random_weights
python ./yolov5/train.py --img $IMG_SIZE --epochs $EPOCHS --data $YAML_FOLD_1 --weights '' --cfg yolov5m.yaml --device $DEVICE --batch-size $BATCH_SIZE --project $PROJECT --save-period $SAVE_PERIOD --name yolov5m-fold_1-three_class_random_weights