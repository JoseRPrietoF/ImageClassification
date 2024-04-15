# combinations=('4950_4952')
models=('convnext_base 2 20') #convnext_base
# models=('resnet18 4 15' 'resnet101 2 20' 'convnext_base 2 20')
width=1024
# batch_size=4
work_dir_all=works/1folder
img_dirs=data/JMBD/1folder
corpus=JMBD
learning_rate=0.0001
# folders=('4946')
folders=('4946' '4949' '4950' '4952')
img_dirs_test=data/JMBD
for folder in "${folders[@]}"; do
for model in "${models[@]}"; do
    set -- $model
    model=$1
    batch_size=$2
    epochs=$3
    img_dirs_fold=${img_dirs}/JMBD${folder}
    exp_name=exp_${model}_size${width}
    work_dir=${work_dir_all}/work_JMBD${folder}_LOO_${model}_size${width}_2
    mkdir -p ${work_dir}
    python main.py --model ${model} --learning_rate ${learning_rate} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir}
    for folder2 in "${folders[@]}"; do
    echo $folder2
    if ! grep -q "$folder2" <<< "$folder" ; then
        checkpoint=${work_dir}/checkpoints/*/*/*/*ckpt
        img_dirs_fold=${img_dirs_test}/JMBD${folder2}
        python main_prod.py --model ${model} --learning_rate ${learning_rate} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${folder2}
    fi
    done
done
done

# cd acts
# ./launch_get_results_1folder.sh
# cd ..