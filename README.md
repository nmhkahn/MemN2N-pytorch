# [WIP] MemN2N-pytorch
PyTorch implementation of [End-To-End Memory Network](https://arxiv.org/abs/1503.08895). This code is heavily based on [memn2n](https://github.com/domluna/memn2n) by domluna.

## Dataset
```shell
cd bAbI
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz
```

## Run
```shell
python memn2n/train.py --task=3 --cuda
```

## Problems (so far)
- So far, implemented only single task scenario
- It seems like model training heavily rely on weight initialization (or might be other things). For example, best performance of task 2 is ~90% however average performance over 100 experiments is ~40% with same model and same hyper-parameters.
