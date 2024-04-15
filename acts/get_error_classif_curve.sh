model=resnet50
folders=(4946 4949 4950 4952)
numdatas=(8 16 32 64 128 256 512)
faux=aux
fauxgt=auxgt
for folder_orig in "${folders[@]}"; do
for numdata in "${numdatas[@]}"; do
    echo "===  Folder tr" ${folder_orig} " model " ${model} " === " ${numdata} " deeds" 
    echo "first line" > $faux
    touch $fauxgt
    te_folders=""
    check=true
    for folder2 in "${folders[@]}"; do
        if ! grep -q "$folder2" <<< "$folder_orig" ; then
            echo "===  Folder te" ${folder2} " === " 
            pathgt=results/JMBD_${folder2}_gt
            path_save_results=/home/jose/projects/image_classif/works/1folder_curve/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_2/results_JMBD${folder2}
            if [ ! -f ${path_save_results} ]; then
                # file does not exist
                path_save_results=/home/jose/projects/image_classif/works/1folder_curve/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_2/results_${folder2}
            fi
            if [ ! -f ${path_save_results} ]; then
                echo "Results for ${folder2} does not exist"
                check=false
            fi
            if [ "$check" = true ] ; then
                tail -n+2 ${path_save_results} >> $faux
                cat ${pathgt} >> $fauxgt
                te_folders="${te_folders}_${folder2}"
            fi
        fi
        
        
    done
    if [ "$check" = true ] ; then
        echo "te folders " ${te_folders}
        python get_error_classif.py --path_hyp ${faux} --path_gt ${fauxgt} --folders ${te_folders}
        rm $faux
        rm $fauxgt
    fi
    
done
done