# folders=(4946 4952)
# models=(resnet18 resnet50 resnet101 convnext_base)
# for folder in "${folders[@]}"; do
# for model in "${models[@]}"; do
# echo "===  Folder " ${folder} " model " ${model} "=== " 
# pathgt=results/JMBD_${folder}_gt
# path_save_results=/home/jose/projects/image_classif/work_JMBD_${folder}_prod_${model}_size1024/results
# python get_error_classif.py --path_hyp ${path_save_results} --path_gt ${pathgt}

# done
# done
# folders=(4946)
# models=(resnet50 convnext_base)
# algs=(raw) 
# for folder in "${folders[@]}"; do
# for model in "${models[@]}"; do
# for alg in "${algs[@]}"; do
# echo "===  Folder " ${folder} " model " ${model} "=== alg" ${alg}  
# pathgt=results/JMBD_${folder}_gt
# path_save_results=results/JMBD_${folder}_${model}_${alg}
# python get_error_classif.py --path_hyp ${path_save_results} --path_gt ${pathgt} --results none
# done
# done
# done

folder_orig=4946
folder=4949
model=resnet50
echo "===  Folder " ${folder} " model " ${model} "=== " 
pathgt=results/JMBD_${folder}_gt
path_save_results=/home/jose/projects/image_classif/works/1folder/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
python get_error_classif.py --path_hyp ${path_save_results} --path_gt ${pathgt}