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

# folder_orig=4946
# folder=4949
# model=resnet50
model=convnext_base
folders=(4946 4949 4950 4952)
faux=aux
fauxgt=auxgt
for folder_orig in "${folders[@]}"; do
    # echo "===  Folder tr" ${folder_orig} " model " ${model} " === " 
    # echo "first line" > $faux
    touch $fauxgt
    te_folders=""
    check=true
    for folder2 in "${folders[@]}"; do
        if ! grep -q "$folder2" <<< "$folder_orig" ; then
            # echo "===  Folder te" ${folder2} " === " 
            pathgt=results/JMBD_${folder2}_gt
            path_save_results=/home/jose/projects/image_classif/works/1folder/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder2}
            if [ ! -f ${path_save_results} ]; then
                # file does not exist
                path_save_results=/home/jose/projects/image_classif/works/1folder/work_JMBD${folder_orig}_LOO_${model}_size1024/results_${folder2}
            fi
            if [ ! -f ${path_save_results} ]; then
                # echo "Results for ${folder2} does not exist"
                check=false
            fi
            if [ "$check" = true ] ; then
                tail -n+2 ${path_save_results} >> $faux
                cat ${pathgt} >> $fauxgt
                te_folders="${te_folders}_${folder2}"
            fi
        fi
        
        
    done
    # only = error cr macro_cr imfcrmcr
    if [ "$check" = true ] ; then
        # echo "te folders " ${te_folders}
        python get_error_classif.py --path_hyp ${faux} --path_gt ${fauxgt} --folders ${te_folders} --only imfcrmcr
        rm $faux
        rm $fauxgt
    fi
    
done