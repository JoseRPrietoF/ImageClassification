# path_seq_save=results/AHPC/1folder_curve/JMBD_tr4952_te4946_resnet50_PD_512_auto
# python get_segmentation_results_AHPC.py --path_hyp ${path_seq_save} --path_gt results/JMBD_4946_gt
# exit
igfile=/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/resultadosIG_tr49.txt
algs=(raw greedy)
num_words=16384
folder_orig=4946
folders=(4949 4950 4952)
# folders=(4949)
# models=('resnet18' 'resnet101' 'convnext_base')
models=('resnet50')
# model=resnet50
path_imgs=/data2/jose/projects/image_classif/data/JMBD
path_works=/data2/jose/projects/image_classif/works/1folder_curve
numdatas=(256)
# numdatas=(8 16 32 64 128 )
names=(auto auto2 auto3 auto4 auto5 auto6 auto7 auto8 auto9 auto10)
# names=(auto10)

for name in "${names[@]}"; do
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_${name}/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
if [ ! -f ${path_res} ]; then
    continue
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}_${numdata}_${name}
path_save_results=results/AHPC/1folder_curve/WeightedSegmentationCost_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} --use_prior_norm True  > ${path_seq_save}
python get_segmentation_results_AHPC.py --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --weighted True --IG_file $igfile --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --calc_seq False > ${path_save_results}
done
done
done
done
done

# numdatas=(8 16 32 64 128 256 )
numdatas=(512)

folder_orig=4949
folders=(4946 4950 4952)
for name in "${names[@]}"; do
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_${name}/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
if [ ! -f ${path_res} ]; then
    continue
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}_${numdata}_${name}
path_save_results=results/AHPC/1folder_curve/WeightedSegmentationCost_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} --use_prior_norm True > ${path_seq_save}
python get_segmentation_results_AHPC.py --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --weighted True --IG_file $igfile --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --calc_seq False > ${path_save_results}
done
done
done
done
done

# numdatas=(8 16 32 64 128 256 )
numdatas=(512)
folder_orig=4950
folders=(4946 4949 4952)
for name in "${names[@]}"; do
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_${name}/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
if [ ! -f ${path_res} ]; then
    continue
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}_${numdata}_${name}
path_save_results=results/AHPC/1folder_curve/WeightedSegmentationCost_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} --use_prior_norm True > ${path_seq_save}
python get_segmentation_results_AHPC.py --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --weighted True --IG_file $igfile --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --calc_seq False > ${path_save_results}
done
done
done
done
done

# numdatas=(8 16 32 64 128 )
numdatas=(256)
folder_orig=4952
folders=(4946 4950 4949)
for name in "${names[@]}"; do
for folder in "${folders[@]}"; do
for alg in "${algs[@]}"; do
for model in "${models[@]}"; do
for numdata in "${numdatas[@]}"; do
path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024_${numdata}_${name}/results_${folder}
if [ ! -f ${path_res} ]; then
    path_res=${path_works}/work_JMBD${folder_orig}_LOO_${model}_size1024/results_JMBD${folder}
fi
if [ ! -f ${path_res} ]; then
    continue
fi
path_img=${path_imgs}/JMBD${folder}/test
path_seq_save=results/AHPC/1folder_curve/JMBD_tr${folder_orig}_te${folder}_${model}_${alg}_${numdata}_${name}
path_save_results=results/AHPC/1folder_curve/WeightedSegmentationCost_tr${folder_orig}_te${folder}_${model}_${alg}2_${numdata}_$name
python sequences_prod.py --model ${model} --folder_orig ${folder_orig} --folder ${folder} --alg ${alg} --GT results/JMBD_${folder_orig}_gt --path_res ${path_res} --path_imgs ${path_img} --use_prior_norm True > ${path_seq_save}
python get_segmentation_results_AHPC.py --path_hyp ${path_seq_save} --path_gt results/JMBD_${folder}_gt --weighted True --IG_file $igfile --path_prix /data/carabela_segmentacion/prod/JMBD_${folder}/ --calc_seq False > ${path_save_results}
done
done
done
done
done