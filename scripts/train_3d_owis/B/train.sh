# !/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# # TRAIN
python main_instance_segmentation.py \
general.OW_task="task1" \
general.split="B" \
general.experiment_name="3D_OWIS" \
general.project_name="3D_OWIS" \
general.use_conf_th=true \
general.margin=3.0 \
general.max_lr=0.0001 \
scheduler=onecyclelr_1stage

python main_instance_segmentation.py \
general.OW_task="task2" \
general.split="B" \
general.experiment_name="3D_OWIS" \
general.project_name="3D_OWIS" \
general.use_conf_th=true \
general.margin=3.0 \
general.max_lr=0.0001 \
scheduler='onecyclelr_1stage' \
general.max_epochs=301

python main_instance_segmentation.py \
general.OW_task="task2" \
general.split="B" \
general.experiment_name="3D_OWIS" \
general.project_name="3D_OWIS" \
general.use_conf_th=true \
general.margin=3.0 \
general.finetune=true \
general.max_lr=0.00001 \
scheduler='onecyclelr_1stage' \
general.max_epochs=101

python main_instance_segmentation.py \
general.OW_task="task3" \
general.split="B" \
general.experiment_name="3D_OWIS" \
general.project_name="3D_OWIS" \
general.use_conf_th=true \
general.margin=3.0 \
general.max_lr=0.0001 \
scheduler='onecyclelr_1stage' \
general.max_epochs=301

python main_instance_segmentation.py \
general.OW_task="task3" \
general.split="B" \
general.experiment_name="3D_OWIS" \
general.project_name="3D_OWIS" \
general.use_conf_th=true \
general.margin=3.0 \
general.finetune=true \
general.max_lr=0.00001 \
scheduler='onecyclelr_1stage' \
general.max_epochs=101