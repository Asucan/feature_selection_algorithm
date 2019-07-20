# feature_selection_algorithm
Some improved feature selection algorithms
1. feature_selection_RFE
由sklearn原始的代码改进而来：https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/feature_selection/rfe.py#L36
  1)递归特征消除数量由当前特征数计算得到，以改善运行速度与递归消除的质量
  2)特征数量小于需求特征数量的2倍时，每次迭代特征消除数量为1，以增加稳定性
