Boston Data Modelling
=====================

.. code:: ipython3

    from sklearn.datasets import load_boston
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    import numpy as np 
    
    np.random.seed(100)
    
    dataset = load_boston()
    
    xTrain, xTest, yTrain, yTest = train_test_split(dataset.data,dataset.target.reshape(-1,1),random_state = 30)
    
    print(xTrain.shape)
    print(xTest.shape)
    dtReg = DecisionTreeRegressor()
    dtModel = dtReg.fit(xTrain,yTrain)
    
    print(dtModel.score(xTrain,yTrain),dtModel.score(xTest,yTest))



.. parsed-literal::

    (379, 13)
    (127, 13)
    1.0 0.8098834820264638

