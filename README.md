# feature_selection_algorithm
## Some improved feature selection algorithms
* feature_selection_RFE
由sklearn原始的代码改进而来：[sklearn.feature_selection.RFE](https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/feature_selection/rfe.py#L36 "悬停显示")
  * 递归特征消除数量由当前特征数计算得到，以改善运行速度与递归消除的质量
  * 特征数量小于需求特征数量的2倍时，每次迭代特征消除数量为1，以增加稳定性
 
* feature_selection_Lasso
  * 原始的lasso算法通过手动改变参数alpha值来进行特征选择
  * 加入改进策略（类似于二分查找。详情请阅读代码），可以通过指定保留特征数量，自主选择alpha值
  * 加入了迭代次数限制，默认100，防止运行时间过长
  * 加入特征数量浮动限制，默认为2，即选择特征数量为10时，最终结果【11，13】都属正常
  * 上述两个限制都可通过参数修改
  * `example`：
    ```python
    from sklearn.datasets import make_friedman1
    x, y = make_friedman1(n_samples=50, n_features=1000, random_state=0)
    print(x.shape)
    slector = lasso(x,y,10)
    slector.run()
    ```
