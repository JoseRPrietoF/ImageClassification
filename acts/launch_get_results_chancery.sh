num_words=16384
# num_words=128
model=RPN
# model=convnext_base
alg=PD
model=101
folder=chancery
path_seq_save=results/chancery/chancery_${alg}_r${model}
path_save_results=results/chancery/BAER_SiSe_${model}_r${model}_${alg}_${num_words}words
path_page_hyp=/data2/jose/projects/RPN_LSTM/works_chancery/work_get_config_mask_rcnn_R_${model}_FPN_1x_giou_1/results/test/inference/page
path_prix=/data/chancery2/labelled_volumes/idxs/all
path_page_gt=/data/chancery2/labelled_volumes/test_page_4classes
python sequences_chancery.py --model ${model} --alg ${alg} --GT results/chancery/train_gt --path_page_hyp ${path_page_hyp} > ${path_seq_save}
python get_results_groups_page_chancery.py --path_prix ${path_prix} --corpus ${folder} --path_hyp ${path_seq_save} --path_gt results/chancery/test_gt --IG_file /data/chancery2/labelled_volumes/words_per_class_train/IG --num_words ${num_words} --path_page_gt ${path_page_gt} --path_page_hyp ${path_page_hyp} --text_hyp no > ${path_save_results}
