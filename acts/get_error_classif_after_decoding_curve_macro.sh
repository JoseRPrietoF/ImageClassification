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
model=resnet50
folders=(4946 4949 4950 4952)
faux=aux
fauxgt=auxgt
decoding=raw2
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
            path_save_results=/home/jose/projects/image_classif/acts/results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder2}_${model}_${decoding}_512_auto*
            for x in `ls ${path_save_results}`; do
                if [ ! -f ${x} ]; then
                    # echo "Results for ${folder2} does not exist"
                    check=false
                fi
                if [ "$check" = true ] ; then
                    tail -n+2 ${x} >> $faux
                    cat ${pathgt} >> $fauxgt
                    te_folders="${te_folders}_${folder2}"
                    # echo $x
                fi
                
            done
        fi
        
        
    done
    if [ "$check" = true ] ; then
        # echo "te folders " ${te_folders}
        python get_error_classif.py --path_hyp ${faux} --path_gt ${fauxgt} --folders ${te_folders} --from_decoder True --only error
        rm $faux
        rm $fauxgt
    fi
    # exit
done