folders=(1576 1577 1578 1579 1580 1581 1582 1583 1584 1585 1586 1588 1589 ) 
# folders=( 1582 ) 
for folder in "${folders[@]}"; do
model=resnet50
alg=PD
path_seq_save=results/AHPC/prod/JMBD_FG_${folder}_${model}_${alg}
path_data=/home/jose/projects/image_classif/works/AHPC_FG_1574/work_FG_${folder}_resnet50_size1024_prod/results_JMBD
path_imgs=/data/carabela_segmentacion/FG_${folder}/test
python sequences_prod.py --path_res ${path_data} --path_imgs ${path_imgs} --model ${model} --folder ${folder} --alg ${alg} --GT results/JMBD_all > ${path_seq_save}
done