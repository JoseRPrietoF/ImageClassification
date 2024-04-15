# folder_xmls=/data2/jose/projects/RPN_LSTM/works_SiSe/work_SiSe_1_chancery_get_config_mask_rcnn_R_50_FPN_1x_giou_acts/results/test/inference/page
# folder_imgs=/data/SimancasSearch/partitions/te/
# path_results="no"
# path_save=SiSe_RPN_hyp_noDecoding
# python show_acts.py --GT False --folder_xmls ${folder_xmls} --folder_imgs ${folder_imgs} --path_results ${path_results} --path_save ${path_save}

# folder_xmls=/data/SimancasSearch/partitions/te_regions/
# folder_imgs=/data/SimancasSearch/partitions/te/
# path_results="no"
# path_save=SiSe_RPN_GT
# python show_acts.py --GT True --folder_xmls ${folder_xmls} --folder_imgs ${folder_imgs} --path_results ${path_results} --path_save ${path_save}


# folder_xmls=/data2/jose/projects/RPN_LSTM/works_SiSe/work_SiSe_1_chancery_get_config_mask_rcnn_R_50_FPN_1x_giou_acts/results/test/inference/page
# folder_imgs=/data/SimancasSearch/partitions/te/
# path_results=/home/jose/projects/image_classif/acts/results/SiSe/pages/JMBD_SiSe_PD
# path_save=SiSe_RPN_hypRPN_PD
# python show_acts.py --GT True --folder_xmls ${folder_xmls} --folder_imgs ${folder_imgs} --path_results ${path_results} --path_save ${path_save}

# path_GT=/home/jose/projects/image_classif/data/EDA/SiSe_RPN_GT
# path_results_1=/home/jose/projects/image_classif/data/EDA/SiSe_RPN_hyp_noDecoding
# path_results_2=/home/jose/projects/image_classif/data/EDA/SiSe_RPN_hypRPN_PD
# path_save=SiSe_RPN_all
# python prepare_imgs.py --path_results_1 ${path_results_1} --path_results_2 ${path_results_2} --path_GT ${path_GT} --path_save ${path_save}

path_GT=/data/SimancasSearch/partitions/te_regions/
folder_xmls=/data2/jose/projects/RPN_LSTM/works_SiSe/work_SiSe_1_chancery_get_config_mask_rcnn_R_50_FPN_1x_giou_acts2/results/test/inference/page
folder_imgs=/data/SimancasSearch/partitions/te/
path_results=/home/jose/projects/image_classif/acts/results/SiSe/pages/JMBD_SiSe_PD
# python show_acts_coco.py --folder_xmls ${folder_xmls} --GT ${path_GT} --path_results ${path_results} --path_images ${folder_imgs}
python show_acts_coco.py --folder_xmls ${folder_xmls} --GT ${path_GT} --path_results ${path_results} --path_images ${folder_imgs} --imfc si