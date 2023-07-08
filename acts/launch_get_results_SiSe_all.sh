num_words=16384
# num_words=128
model=resnet50
# model=convnext_base
alg=PD
folder=SiSe
path_seq_save=results/SiSe/all/JMBD_SiSe_${model}_${alg}
path_save_results=results/SiSe/all/BAER_SiSe_${model}_${alg}
python sequences_SiSe_all.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt > ${path_seq_save}
python get_results_groups.py --corpus ${folder} --path_prix /home/jose/projects/image_classif/data/SiSe/all_text/test --path_hyp ${path_seq_save} --path_gt results/SiSe/test_all_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt --num_words ${num_words} > ${path_save_results}
