# MemN2N-pytorch
PyTorch implementation of [End-To-End Memory Network](https://arxiv.org/abs/1503.08895). This code is heavily based on [memn2n](https://github.com/domluna/memn2n) by domluna.

## Dataset
```shell
cd bAbI
wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
tar xzvf ./tasks_1-20_v1-2.tar.gz
```

## Training
```shell
python memn2n/train.py --task=3 --cuda
```

## Results (single-task only)
In all experiments, hyperparameters follow the settings in `memn2n/train.py` (e.g. lr=0.001). 

And since I suspect training is really unstable, I train the model 100 times in each task with fixed hyperparameters described in `memn2n/train.py`, then average top-5 results.

Task  |  Training Acc.  |  Test Acc.  |  Pass
------|-----------------|-------------|--------
1     |  1.00           |  1.00       |    O
2     |  0.98           |  0.84       |                 
3     |  1.00           |  0.49       |                  
4     |  1.00           |  0.99       |    O             
5     |  1.00           |  0.94       |                 
6     |  1.00           |  0.93       |                  
7     |  0.96           |  0.95       |   O              
8     |  0.97           |  0.89       |                 
9     |  1.00           |  0.91       |                 
10    |  1.00           |  0.87       |                 
11    |  1.00           |  0.98       |   O              
12    |  1.00           |  1.00       |   O              
13    |  0.97           |  0.94       |                 
14    |  1.00           |  1.00       |   O              
15    |  1.00           |  1.00       |   O              
16    |  0.81           |  0.47       |                 
17    |  0.75           |  0.53       |                
18    |  0.97           |  0.92       |                 
19    |  0.39           |  0.17       |                 
20    |  1.00           |  1.00       |   O     
mean  |  0.94           |  0.84       |

## Issues
- It seems like model training heavily rely on weight initialization (or training is very unstable). For example, best performance of task 2 is ~90% however average performance over 100 experiments is ~40% with same model and same hyperparameters.
- WHY?

## TODO
- Multi-task learning
