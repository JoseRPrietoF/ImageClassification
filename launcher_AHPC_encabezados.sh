# models=('convnext_base') #convnext_base
# for model in "${models[@]}"; do
#     python main.py "SiSe/cut" "SiSe" ${model}
# done
work_dir_all=works/AHPC_encabezados/
width=512
models=('resnet50') #convnext_base
corpus=AHPC_encabezados
img_dirs=data/AHPC_cabeceras
batch_size=4
epochs=30
for model in "${models[@]}"; do
    exp_name=exp_${model}_size${width}
    work_dir=${work_dir_all}work_${model}_size${width}
    mkdir -p ${work_dir}
    python main.py --model ${model} --learning_rate 0.001 --batch_size ${batch_size} --epochs ${epochs} --corpus ${corpus} --width ${width} --img_dirs ${img_dirs} --n_classes 12 --work_dir ${work_dir} --num_input_channels 3
done