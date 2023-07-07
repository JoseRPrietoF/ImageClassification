combinations=('4946_4949' '4946_4950' '4946_4952' '4949_4950' '4949_4952' '4950_4952')
models=('resnet50') #convnext_base
for combination in "${combinations[@]}"; do
for model in "${models[@]}"; do
    name=JMBD_tr_${combination}
    path_data=JMBD/2folders/${name}
    ls data/JMBD/2folders/JMBD_tr_${combination}
    python main.py ${path_data} ${name} ${model}
done
done
# python main.py "JMBD/2folders/JMBD_tr_4946_4949" "JMBD_tr_4946_4949" "resnet50"


# python main_prod.py "4946_4949" "data/JMBD/JMBD4950" "JMBD4950"
# python main_prod.py "4946_4949" "data/JMBD/JMBD4952" "JMBD4952"


# python main_prod.py "4946_4950" "data/JMBD/JMBD4946" "JMBD4946"
# python main_prod.py "4946_4950" "data/JMBD/JMBD4950" "JMBD4950"


# python main_prod.py "JMBD4950" "data/JMBD/JMBD4946" "JMBD4946"
# python main_prod.py "JMBD4950" "data/JMBD/JMBD4949" "JMBD4949"


# python main_prod.py "JMBD4952" "data/JMBD/JMBD4946" "JMBD4946"
# python main_prod.py "JMBD4952" "data/JMBD/JMBD4949" "JMBD4949"
