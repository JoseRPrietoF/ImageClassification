# model=resnet50
model=convnext_base
alg=PD
folder=SiSe
path_seq_save=results/SiSe/JMBD_SiSe_${model}_${alg}
path_save_results=results/SiSe/BAER_SiSe_${model}_${alg}
python sequences_SiSe.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt > ${path_seq_save}
python get_results.py --corpus ${folder} --path_prix /home/jose/projects/image_classif/data/SiSe/cut_text/test --path_hyp ${path_seq_save} --path_gt results/SiSe/test_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt > ${path_save_results}
