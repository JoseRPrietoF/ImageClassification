num_words=16384
# num_words=128
model=RPN
# model=convnext_base
alg=PD
folder=SiSe
path_seq_save=results/SiSe/pages/JMBD_SiSe_${alg}101
path_save_results=results/SiSe/pages/BAER_SiSe_${model}_${alg}101
path_page_hyp=/data2/jose/projects/RPN_LSTM/works_SiSe/work_SiSe_1_chancery_get_config_mask_rcnn_R_50_FPN_1x_giou_acts101/results/test/inference/page
path_page_gt=/data/SimancasSearch/partitions/te_regions
python sequences_SiSe_all_pages.py --model ${model} --folder ${folder} --alg ${alg} --GT results/SiSe/test_gt --path_page_hyp ${path_page_hyp} > ${path_seq_save}
python get_results_groups_page.py --corpus ${folder} --path_hyp ${path_seq_save} --path_gt results/SiSe/test_all_gt --IG_file /home/jose/projects/docClasifIbPRIA22/data/SiSe/IG_TFIDF/train/resultadosIG_train.txt --num_words ${num_words} --path_page_gt ${path_page_gt} --path_page_hyp ${path_page_hyp} --text_hyp no > ${path_save_results}