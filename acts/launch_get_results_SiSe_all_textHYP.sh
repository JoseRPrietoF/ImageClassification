num_words=16384
# num_words=128
model=resnet50
# model=convnext_base
alg=PD
folder=SiSe
path_seq_save=results/SiSe/all/JMBD_SiSe_${model}_${alg}_textHYP
path_save_results=results/SiSe/all/BAER_SiSe_${model}_${alg}_textHYP
path_res=/home/jose/projects/image_classif/works/SiSe/all/work_resnet50_size1024/results
path_imgs=/home/jose/projects/image_classif/data/SiSe/all/test
path_page_hyp_text=/home/jose/SiSe/Sise_test_p2pala_trained/PAGE-TRANSCRIPTS
path_page_gt=/data/SimancasSearch/partitions/te_regions
# python sequences_SiSe_all.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt --path_imgs ${path_imgs} --path_res ${path_res} > ${path_seq_save}
python get_results_groups.py --corpus ${folder} --path_page_gt ${path_page_gt} --path_hyp ${path_seq_save} --path_gt results/SiSe/test_all_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt --num_words ${num_words} --text_hyp ${path_page_hyp_text} > ${path_save_results}
