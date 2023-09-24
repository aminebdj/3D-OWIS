# !/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# # TRAIN
python main_instance_segmentation.py \
general.OW_task="task1" \
general.split="C" \
general.experiment_name="3D_OWIS-PC-CT" \
general.project_name="3D_OWIS" \
general.use_conf_th=false \
general.margin=1.0 \
general.max_lr=0.0001

python main_instance_segmentation.py \
general.OW_task="task2" \
general.split="C" \
general.experiment_name="3D_OWIS-PC-CT" \
general.project_name="3D_OWIS" \
general.use_conf_th=false \
general.margin=1.0 \
general.max_lr=0.0001

python main_instance_segmentation.py \
general.OW_task="task2" \
general.split="C" \
general.experiment_name="3D_OWIS-PC-CT" \
general.project_name="3D_OWIS" \
general.use_conf_th=false \
general.margin=1.0 \
general.finetune=true \
general.max_lr=0.0001

python main_instance_segmentation.py \
general.OW_task="task3" \
general.split="C" \
general.experiment_name="3D_OWIS-PC-CT" \
general.project_name="3D_OWIS" \
general.use_conf_th=false \
general.margin=1.0 \
general.max_lr=0.0001

python main_instance_segmentation.py \
general.OW_task="task3" \
general.split="C" \
general.experiment_name="3D_OWIS-PC-CT" \
general.project_name="3D_OWIS" \
general.use_conf_th=false \
general.margin=1.0 \
general.finetune=true \
general.max_lr=0.00001