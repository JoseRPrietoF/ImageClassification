folder_orig=4946
model=resnet50
folder=1574
alg=PD
path_seq_save=results/AHPC/prod/JMBD_FG_${folder}_${model}_${alg}
path_data=/data2/jose/projects/image_classif/works/AHPC_FG_1574/work_FG_1574_resnet50_size1024_prod/results_JMBD
path_imgs=/data/carabela_segmentacion/FG_1574/test
python sequences_prod.py --path_res ${path_data} --path_imgs ${path_imgs} --model ${model} --folder ${folder} --alg ${alg} --GT results/JMBD_all > ${path_seq_save}
