# combinations=('4946_4949' '4946_4950' '4946_4952' '4949_4950' '4949_4952' '4950_4952')
combinations=('4950_4952')
models=('resnet50') #convnext_base
width=1024
batch_size=4
epochs=15
work_dir_all=works/2folder
img_dirs=data/JMBD/2folders
corpus=JMBD
folders=('4946' '4949' '4950' '4952')
img_dirs_test=data/JMBD
for combination in "${combinations[@]}"; do
for model in "${models[@]}"; do
    img_dirs_fold=${img_dirs}/JMBD_tr_${combination}
    exp_name=exp_${model}_size${width}
    work_dir=${work_dir_all}/work_JMBD_tr_${combination}_${model}_size${width}
    mkdir -p ${work_dir}
    python main.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir}
    for folder in "${folders[@]}"; do
    if ! grep -q "$folder" <<< "$combination" ; then
        checkpoint=${work_dir}/checkpoints/*/*/*/*ckpt
        img_dirs_fold=${img_dirs_test}/JMBD${folder}
        python main_prod.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${folder}
    fi
    done
done
done

cd acts
./launch_get_results_2folder.sh