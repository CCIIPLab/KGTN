# Knowledge Enhanced Multi-intent Transformer Network for Recommendation

This is our Pytorch implementation for the paper:
> Ding Zou, Wei Wei, Feida Zhu, Chuanyu Xu, Tao Zhang, Chengfu Huo (2024). Knowledge Enhanced Multi-intent Transformer Network for Recommendation , [Paper in arXiv](https://arxiv.org/). In WWW 2024 (Industry Track).


## Requirement
The code has been tested running under Python 3.7.9. The required packages are as follows:
- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- torch_sparse == 0.6.10
- networkx == 2.5

## Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes (see the parser function in utils/parser.py).
* Train and Test

```
python main.py 
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{
  kgtn2024,
  title={Knowledge Enhanced Multi-intent Transformer Network for Recommendation},
  author={Zou, Ding and Wei, Wei and Zhu, Feida and Xu, Chuanyu and Zhang, Tao and Huo, Chengfu},
  booktitle={Proceedings of the ACM Web Conference 2024},
  year={2024}
}

```



## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)." to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,967   |    2,445  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |      42,346|
|    Knowledge Graph    | #Entities     |      77,903   |    182,011|      9,366 |
|                       | #Relations    |          25   |         12|         60 |
|                       | #Triplets     |   151,500     |  1,241,996|     15,518 |


## Reference 
- We partially use the codes of [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network).
- You could find all other baselines in Github.
