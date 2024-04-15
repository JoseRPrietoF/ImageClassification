# combinations=('4950_4952')
# models=('convnext_base 2 20') #convnext_base
# models=('resnet18 4 15' 'resnet101 2 20' 'convnext_base 2 20')
models=('resnet50 4 100' )
width=1024
# batch_size=4
work_dir_all=works/1folder_curve
img_dirs=/data/AHPC_book_segm/1folder
corpus=JMBD
learning_rate=0.001
# folders=('4946')
folders=('4946' '4949' '4950' '4952')
numdatas=(512)
# numdatas=(8 16 32 64 128 256 512)
exps=(auto auto2 auto3 auto4 auto5 auto6 auto7 auto8 auto9 auto10)
img_dirs_test=data/JMBD
model=resnet50
batch_size=4
epochs=50
for exp in "${exps[@]}"; do
for folder in "${folders[@]}"; do
# for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
    # set -- $model
    # model=$1
    # batch_size=$2
    # epochs=$3
    img_dirs_fold=${img_dirs}/curve/JMBD${folder}_${numdata}
    exp_name=exp_${model}_size${width}_${numdata}
    work_dir=${work_dir_all}/work_JMBD${folder}_LOO_${model}_size${width}_${numdata}_${exp}
    mkdir -p ${work_dir}
    echo ${img_dirs_fold}
    # python main.py --model ${model} --learning_rate ${learning_rate} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --auto_lr_find true --patience 5
    for folder2 in "${folders[@]}"; do
    echo $folder2
    if ! grep -q "$folder2" <<< "$folder" ; then
        checkpoint=${work_dir}/checkpoints/*/*/*/*ckpt
        img_dirs_fold=${img_dirs_test}/JMBD${folder2}
        # python main_prod.py --model ${model} --learning_rate ${learning_rate} --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs_fold} --n_classes 3 --work_dir ${work_dir} --checkpoint_load ${checkpoint} --output_name ${folder2}
    fi
    done
done
# done
done
cd acts
./launch_get_results_1folder_curve.sh $exp #&
cd ..
done
