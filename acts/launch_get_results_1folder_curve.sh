# --num_words 128
# name=$1
name=auto
algs=(PD)
num_words=16384
folder_orig=4946
folders=(4949 4950 4952)
# models=('resnet18' 'resnet101' 'convnext_base')
models=('resnet50')
# model=resnet50
path_imgs=/data2/jose/projects/image_classif/data/JMBD
path_works=/data2/jose/projects/image_classif/works/1folder_curve
numdatas=(512)
# numdatas=(8 16 32 64 128 256 512)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_${name}/results_${folder}
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_${name}
path_save_results=results/AHPC/1folder_curve/BAER_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done
done

folder_orig=4949
folders=(4946 4950 4952)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_$name/results_${folder}
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
path_save_results=results/AHPC/1folder_curve/BAER_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done
done

folder_orig=4950
folders=(4949 4946 4952)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_$name/results_${folder}
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
path_save_results=results/AHPC/1folder_curve/BAER_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done
done

folder_orig=4952
folders=(4949 4946 4950)
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_$name/results_${folder}
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
path_save_results=results/AHPC/1folder_curve/BAER_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} > ${path_seq_save}
python get_results.py --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --IG_file /data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt --num_words ${num_words} > ${path_save_results}
done
done
done
done
