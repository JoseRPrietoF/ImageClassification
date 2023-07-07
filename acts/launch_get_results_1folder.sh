# --num_words 128
algs=(PD greedy)
num_words=16384
folder_orig=4946
folders=(4949 4950 4952)
model=resnet50
# for folder in "${folders[@]}"; do
# for alg in "${algs[@]}"; do
# path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
# path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
# python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt > ${path_seq_save}
# python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
# done
# done

folder_orig=4949
folders=(4946 4950 4952)
model=resnet50
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done

folder_orig=4950
folders=(4949 4946 4952)
model=resnet50
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done

folder_orig=4952
folders=(4949 4946 4950)
model=resnet50
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
path_seq_save=results/AHPC/1folder/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}
path_save_results=results/AHPC/1folder/BAER_tr${folder_orig}_te${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done