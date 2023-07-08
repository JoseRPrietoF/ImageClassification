model=resnet50
# model=convnext_base
alg=PD
folder=SiSe
path_seq_save=results/SiSe/JMBD_SiSe_${model}_${alg}_text
path_save_results=results/SiSe/BAER_SiSe_${model}_${alg}_text   
path_res=/home/jose/projects/docClasifIbPRIA22/works_SiSe/cut/work__128,128_numFeat1024/results.txt
path_imgs=/home/jose/projects/image_classif/data/SiSe/cut/test
python sequences_SiSe.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt --path_res ${path_res} --path_imgs ${path_imgs} > ${path_seq_save}
python get_results.py --corpus ${folder} --path_prix /home/jose/projects/image_classif/data/SiSe/cut_text/test --path_hyp ${path_seq_save} --path_gt results/SiSe/test_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt > ${path_save_results}

# all
path_seq_save=results/SiSe/all/JMBD_SiSe_${model}_${alg}_text
path_save_results=results/SiSe/all/BAER_SiSe_${model}_${alg}_text   
path_res=/home/jose/projects/docClasifIbPRIA22/works_SiSe/all/work__128,128_numFeat1024/results.txt
path_imgs=/home/jose/projects/image_classif/data/SiSe/all/test
python sequences_SiSe_all.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt --path_res ${path_res} --path_imgs ${path_imgs} > ${path_seq_save}
python get_results_groups.py --corpus ${folder} --path_prix /home/jose/projects/image_classif/data/SiSe/all_text/test/ --path_hyp ${path_seq_save} --path_gt results/SiSe/test_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt > ${path_save_results}