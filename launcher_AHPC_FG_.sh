model=('resnet50 4 20')
width=1024
work_dir_all=works/JMBD_ALL
img_dirs_fold=data/JMBD/all_JMBD
corpus=JMBD
set -- $model
model=$1
batch_size=$2
epochs=$3
exp_name=exp_${model}_size${width}
work_dir_tr=${work_dir_all}/work_JMBD_tr_all_${model}_size${width}
mkdir -p ${work_dir_tr}
# python main.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir_tr}

folders=( 1580 ) #convnext_base
work_dir_all=works/AHPC_FG_1574
corpus=JMBD
for folder in "${folders[@]}"; do
echo ">>>>>>>> FOLDER ${folder}"
work_dir=${work_dir_all}/work_FG_${folder}_${model}_size${width}_prod
mkdir -p ${work_dir}

# checkpoint=works/FG_1574/work_FG_1574_tr_all_resnet50_size1024_corrected/checkpoints/work/0/checkpoints/epoch=19-step=8619.ckpt
checkpoint=works/FG_1574/work_FG_1574_tr_all_resnet50_size1024_correctedDA/checkpoints/work/0/checkpoints/epoch=19-step=8619.ckpt
img_dirs_fold=/data/carabela_segmentacion/FG_${folder}
python main_prod.py --model ${model} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${corpus}
done

# work_dir_all=works/AHPC_FG_1574
# corpus=JMBD
# work_dir=${work_dir_all}/work_FG_1574_${model}_size${width}_prod
# mkdir -p ${work_dir}

# checkpoint=${work_dir_tr}/checkpoints/*/*/*/*ckpt
# img_dirs_fold=/data/carabela_segmentacion/FG_1574
# python main_prod.py --model ${model} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${corpus}


