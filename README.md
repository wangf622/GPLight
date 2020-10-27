# GPLight

GPLight (See the [paper](https://arxiv.org/abs/2009.14627) in arXiv) is a deep reinforcement learning agent for multi-intersection traffic signal control with GNN predicting future traffic.

We first predict the future traffic throughput is predicted using the [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) and then control the traffic signal based on the prediction and the current traffic condition. The code follows similat structure [CoLight](https://github.com/wingsweihua/colight).
