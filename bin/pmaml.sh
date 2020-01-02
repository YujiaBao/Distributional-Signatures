#dataset=amazon
#data_path='data/amazon.json'
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=huffpost
#data_path='data/huffpost.json'
#n_train_class=20
#n_val_class=5
#n_test_class=16

dataset=reuters
data_path='data/reuters.json'
n_train_class=15
n_val_class=5
n_test_class=11

pretrained_bert='bert-base-uncased'
# For P-MAML, use need to replace pretrained_bert by the output of
# https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py
# Otherwise, MAML will be learned from bert_base_uncased
bert_cache_dir='~/.pytorch_pretrained_bert/'

python src/main.py \
    --cuda 0 \
    --way 5 \
    --shot 1 \
    --query 25 \
    --mode 'train' \
    --embedding ebd \
    --classifier 'mlp' \
    --mlp_hidden 5 \
    --finetune_ebd \
    --maml \
    --maml_innersteps 10 \
    --maml_firstorder \
    --maml_stepsize 0.001 \
    --bert \
    --pretrained_bert $pretrained_bert \
    --bert_cache_dir $bert_cache_dir \
    --lr 0.00001 \
    --dataset=$dataset \
    --data_path=$data_path \
    --n_train_class=$n_train_class \
    --n_val_class=$n_val_class \
    --n_test_class=$n_test_class
