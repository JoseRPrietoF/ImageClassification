# models=('convnext_base') #convnext_base
# for model in "${models[@]}"; do
#     python main.py "SiSe/cut" "SiSe" ${model}
# done
work_dir_all=works/SiSe/all/
width=1024
models=('resnet50') #convnext_base
corpus=SiSe
img_dirs=data/SiSe/all
for model in "${models[@]}"; do
    batch_size=4
    epochs=30
    exp_name=exp_${model}_size${width}
    work_dir=${work_dir_all}work_${model}_size${width}
    mkdir -p ${work_dir}
    python main.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs} --n_classes 4 --work_dir
done