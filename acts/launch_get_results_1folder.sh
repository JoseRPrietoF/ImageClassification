# --num_words 128
algs=(raw)
num_words=16384
# num_words=128
folder_orig=4946
folders=(4949 4950 4952)
# folders=(4949)
models=('resnet50' 'resnet18' 'resnet101' 'convnext_base')
# models=('convnext_base')
# model=resnet50
path_imgs=/data2/jose/projects/image_classif/data/JMBD
path_works=/data2/jose/projects/image_classif/works/1folder
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done

exit

folder_orig=4949
folders=(4946 4950 4952)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done

folder_orig=4950
folders=(4949 4946 4952)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done

folder_orig=4952
folders=(4949 4946 4950)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done

