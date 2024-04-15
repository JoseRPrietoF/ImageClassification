num_words=16384
# num_words=128
model=RPN
# model=convnext_base
alg=PD
folder=SiSe

path_seq_save=results/SiSe/pages/JMBD_SiSe_classifFromRPN_${alg}_textHYP101
path_save_results=results/SiSe/pages/BAER_SiSe_classifFromRPN_${model}_${alg}_textHYP101

path_page_hyp=/data2/jose/projects/RPN_LSTM/works_SiSe/work_SiSe_1_chancery_get_config_mask_rcnn_R_50_FPN_1x_giou_acts101/results/test/inference/page
path_page_hyp_text=/home/jose/SiSe/Sise_test_p2pala_trained/PAGE-TRANSCRIPTS
path_page_gt=/data/SimancasSearch/partitions/te_regions
path_classif=/home/jose/projects/image_classif/works/SiSe/all/work_resnet50_size1024/results_SiSe_RPN
python sequences_SiSe_all_pages.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt --path_page_hyp ${path_page_hyp} --from_classif ${path_classif} > ${path_seq_save} 
python get_results_groups_page.py --corpus ${folder} --path_hyp ${path_seq_save} --path_gt results/SiSe/test_all_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt --num_words ${num_words} --path_page_gt ${path_page_gt} --path_page_hyp ${path_page_hyp} --text_hyp ${path_page_hyp_text} > ${path_save_results}
