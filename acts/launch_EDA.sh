# folder=4946
# alg=greedy
# model=resnet18
# folder=$1
# alg=$2
# model=$3
folders=(4946 4949 4950 4952)
# models=(resnet50)
# algs=(PD)
for folder in "${folders[@]}"; do
echo "===  Folder " ${folder} "=== " 
path_seq_save=results/JMBD_${folder}_${model}_${alg}
path_save_results=results/BAER_${folder}_${model}_${alg}
python EDA.py --path_gt results/JMBD_${folder}_gt
done