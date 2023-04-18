# folder=$1
# model=$2
folders=(4952)
models=(convnext_base)
for folder in "${folders[@]}"; do
for model in "${models[@]}"; do
path_res=/data2/jose/projects/image_classif/work_JMBD_${folder}_prod_${model}_size1024/results
python compute_incosistencies.py --path ${path_res} > results/incosist_JMBD${folder}_${model}
done
done