# folder=4946
# alg=greedy
# model=resnet18
# folder=$1
# alg=$2
# model=$3
folders=(4946 4952)
models=(resnet18 resnet50 resnet101 convnext_base )
algs=(PD)
for folder in "${folders[@]}"; do
for model in "${models[@]}"; do
for alg in "${algs[@]}"; do
path_seq_save=results/JMBD_${folder}_${model}_${alg}
path_save_results=results/BAER_${folder}_${model}_${alg}
python sequences_prod.py --model ${model} --folder ${folder} --alg ${alg} --GT results/JMBD_4949_4950_gt > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt > ${path_save_results}
done
done
done
# folders=(4952)
# models=( resnet50 convnext_base )
# algs=(raw)
# for folder in "${folders[@]}"; do
# for model in "${models[@]}"; do
# for alg in "${algs[@]}"; do
# path_seq_save=results/JMBD_${folder}_${model}_${alg}
# path_save_results=results/BAER_${folder}_${model}_${alg}
# python sequences_prod.py --model ${model} --folder ${folder} --alg ${alg} > ${path_seq_save}
# python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --alg ${alg} --class_to_cut I  #> ${path_save_results}
# done
# done
# done
# --num_words 128