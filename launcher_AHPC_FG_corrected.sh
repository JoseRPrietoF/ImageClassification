model=('resnet50 4 20')
width=1024
work_dir_all=works/FG_1574
img_dirs_fold=/data/carabela_segmentacion/FG_1574_gt
corpus=JMBD
set -- $model
model=$1
batch_size=$2
epochs=$3
exp_name=exp_${model}_size${width}
work_dir_tr=${work_dir_all}/work_FG_1574_tr_all_${model}_size${width}_correctedDA
checkpoint_load=/home/jose/projects/image_classif/works/JMBD_ALL/work_JMBD_tr_all_resnet50_size1024/checkpoints/work/0/checkpoints/epoch=19-step=27179.ckpt
mkdir -p ${work_dir_tr}
python main.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir_tr} --checkpoint_load ${checkpoint_load}

# work_dir_all=works/AHPC_FG_1574
# corpus=JMBD
# work_dir=${work_dir_all}/work_FG_1575_${model}_size${width}_prod
# mkdir -p ${work_dir}

# checkpoint=works/FG_1574/work_FG_1574_tr_all_resnet50_size1024_corrected/checkpoints/work/0/checkpoints/epoch=19-step=8619.ckpt
# img_dirs_fold=/data/carabela_segmentacion/FG_1575
# python main_prod.py --model ${model} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${corpus}