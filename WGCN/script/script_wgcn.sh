# REPRO
## TransE
python run.py --gpu 0 --name repro --decoder transe --n-layer 1 --init_emb_size 200 -fa --data FB15k-237

## DistMult
python run.py --gpu 0 --name repro --decoder distmult -fb -fa -fd --data FB15k-237

## ConvE
python run.py --gpu 0 --name repro --decoder conve -fb -fa -fd --data FB15k-237


