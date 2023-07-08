# --num_words 128
# combinations=('4946_4949' '4946_4950' '4946_4952' '4949_4950' '4949_4952' '4950_4952')
combinations=('4950_4952')
algs=(PD)
num_words=16384 # 16384
folders=('4946' '4949' '4950' '4952')
model=resnet50
path_works=/data2/jose/projects/image_classif/works/2folder
path_imgs=/data2/jose/projects/image_classif/data/JMBD
for combination in "${combinations[@]}"; do
for alg in "${algs[@]}"; do
for folder in "${folders[@]}"; do
if ! grep -q "$folder" <<< "$combination" ; then
path_res=${path_works}/work_JMBD_tr_${combination}_${model}_size1024/results_${folder}
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/2folder/JMBD_tr${combination}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/2folder/BAER_tr${combination}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}

fi
done
done
done
