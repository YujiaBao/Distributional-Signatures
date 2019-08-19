#dataset=amazon
#data_path="data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path="data/fewrel.json"
#n_train_class=65
#n_val_class=5
#n_test_class=10

#dataset=20newsgroup
#data_path="data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7

#dataset=huffpost
#data_path="data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16

#dataset=rcv1
#data_path="data/rcv1.json"
#n_train_class=37
#n_val_class=10
#n_test_class=24

dataset=reuters
data_path="data/reuters.json"
n_train_class=15
n_val_class=5
n_test_class=11

if [ "$dataset" = "fewrel" ]; then
    python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --auxiliary pos \
        --meta_iwf \
        --meta_w_target
else
    python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --meta_iwf \
        --meta_w_target
fi
