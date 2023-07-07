models=('resnet50') #convnext_base
for model in "${models[@]}"; do
    # python main.py "JMBD/JMBD4946" "JMBD4946_LOO" ${model}
    python main.py "JMBD/JMBD4949" "JMBD4949_LOO" ${model}
    python main.py "JMBD/JMBD4950" "JMBD4950_LOO" ${model}
    python main.py "JMBD/JMBD4952" "JMBD4952_LOO" ${model}
done
##JMBD4946
# python main_prod.py "JMBD4946" "data/JMBD/JMBD4949" "JMBD4949"
# python main_prod.py "JMBD4946" "data/JMBD/JMBD4950" "JMBD4950"
# python main_prod.py "JMBD4946" "data/JMBD/JMBD4952" "JMBD4952"

##JMBD4949
python main_prod.py "JMBD4949" "data/JMBD/JMBD4946" "JMBD4946"
python main_prod.py "JMBD4949" "data/JMBD/JMBD4950" "JMBD4950"
python main_prod.py "JMBD4949" "data/JMBD/JMBD4952" "JMBD4952"

# ##JMBD4950
python main_prod.py "JMBD4950" "data/JMBD/JMBD4946" "JMBD4946"
python main_prod.py "JMBD4950" "data/JMBD/JMBD4949" "JMBD4949"
python main_prod.py "JMBD4950" "data/JMBD/JMBD4952" "JMBD4952"

# ##JMBD4952
python main_prod.py "JMBD4952" "data/JMBD/JMBD4946" "JMBD4946"
python main_prod.py "JMBD4952" "data/JMBD/JMBD4949" "JMBD4949"
python main_prod.py "JMBD4952" "data/JMBD/JMBD4950" "JMBD4950"