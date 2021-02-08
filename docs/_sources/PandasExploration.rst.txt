Pandas Exploration
==================

.. code:: ipython3

    import pandas as pd
    import numpy as np 
    from sklearn.datasets import load_iris 

.. code:: ipython3

    iris = load_iris()

.. code:: ipython3

    featureColumns = [i.replace(" ","").replace("(cm)","") for i in iris.feature_names]
    df = pd.DataFrame(iris.data,columns=featureColumns)
    df['target'] = iris.target
    
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>target</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>5.1</td>
          <td>3.5</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5.0</td>
          <td>3.6</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    ## Single filter
    df[df['sepallength'] < 5].head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>target</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>4.6</td>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
          <td>0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>4.4</td>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    ## applying 2 filters 
    
    df[(df['sepallength'] < 5) & (df['target'].isin([0,1]))].head()





.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>sepallength</th>
          <th>sepalwidth</th>
          <th>petallength</th>
          <th>petalwidth</th>
          <th>target</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1</th>
          <td>4.9</td>
          <td>3.0</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.7</td>
          <td>3.2</td>
          <td>1.3</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4.6</td>
          <td>3.1</td>
          <td>1.5</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>4.6</td>
          <td>3.4</td>
          <td>1.4</td>
          <td>0.3</td>
          <td>0</td>
        </tr>
        <tr>
          <th>8</th>
          <td>4.4</td>
          <td>2.9</td>
          <td>1.4</td>
          <td>0.2</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



Transforming Data
-----------------

.. code:: ipython3

    df = pd.DataFrame({
        'temperature' : pd.Series(23 + 10*np.random.randn(11)),
        'thunderstorm' : pd.Series(150 + 10*np.random.randn(11)),
        'location' : list('XXYYXXYYXXY')
        
    })
    
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temperature</th>
          <th>thunderstorm</th>
          <th>location</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>37.178703</td>
          <td>151.250130</td>
          <td>X</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.338412</td>
          <td>148.930679</td>
          <td>X</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.614139</td>
          <td>151.664537</td>
          <td>Y</td>
        </tr>
        <tr>
          <th>3</th>
          <td>35.818557</td>
          <td>154.044738</td>
          <td>Y</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.598561</td>
          <td>143.369174</td>
          <td>X</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    replaceValues = {
        'location' : {
            "X" : "MISSISSIPPI",
            "Y" : "MANALI"
        }
    }
    
    df = df.replace(replaceValues,regex=True)
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temperature</th>
          <th>thunderstorm</th>
          <th>location</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>37.178703</td>
          <td>151.250130</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.338412</td>
          <td>148.930679</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>2</th>
          <td>16.614139</td>
          <td>151.664537</td>
          <td>MANALI</td>
        </tr>
        <tr>
          <th>3</th>
          <td>35.818557</td>
          <td>154.044738</td>
          <td>MANALI</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.598561</td>
          <td>143.369174</td>
          <td>MISSISSIPPI</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # df.location.str.contains("ISSI")
    df.loc[df.location.str.contains("ISSI")]




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temperature</th>
          <th>thunderstorm</th>
          <th>location</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>37.178703</td>
          <td>151.250130</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>1</th>
          <td>16.338412</td>
          <td>148.930679</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>4</th>
          <td>21.598561</td>
          <td>143.369174</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>5</th>
          <td>29.470110</td>
          <td>141.694351</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>8</th>
          <td>35.765885</td>
          <td>144.513669</td>
          <td>MISSISSIPPI</td>
        </tr>
        <tr>
          <th>9</th>
          <td>27.894740</td>
          <td>156.470016</td>
          <td>MISSISSIPPI</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.groupby('location').mean()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temperature</th>
          <th>thunderstorm</th>
        </tr>
        <tr>
          <th>location</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MANALI</th>
          <td>29.238917</td>
          <td>155.537351</td>
        </tr>
        <tr>
          <th>MISSISSIPPI</th>
          <td>28.041068</td>
          <td>147.704670</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    import pandas as pd 
    import numpy as np 
    
    serIndex = ['s1','s2','s3','s4','s5']
    
    heights_A = pd.Series(np.array([176.2, 158.4, 167.6, 156.2,161.4]),index=serIndex)
    weights_A = pd.Series(np.array([85.1, 90.2, 76.8, 80.4,78.9]),index=serIndex)
    
    
    df_A = pd.DataFrame()
    df_A['Student_height'] = heights_A
    df_A['Student_weight'] = weights_A
    
    df_A['Gender'] = ['M','F','M','M','F']
    
    s = pd.Series(np.array([165.4, 82.7, 'F']),index=['Student_height', 'Student_weight', 'Gender'])
    s.name = 's6'
    df_AA = df_A.append(s)
    # print(df_AA)
    
    np.random.seed(100)
    
    
    heights_B = pd.Series(np.random.normal(loc=170.0,scale=25,size=5))
    
    np.random.seed(100)
    
    weights_B = pd.Series(np.random.normal(loc=75.0,scale=12.0,size=5))
    
    df_B = pd.DataFrame()
    df_B['Student_height'] = heights_B
    df_B['Student_weight'] = weights_B
    df_B.index = ['s7','s8','s9','s10','s11']
    
    df_B['Gender'] = ['F','M','F','F','M']
    
    pd.concat([df_AA,df_B])




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Student_height</th>
          <th>Student_weight</th>
          <th>Gender</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>s1</th>
          <td>176.2</td>
          <td>85.1</td>
          <td>M</td>
        </tr>
        <tr>
          <th>s2</th>
          <td>158.4</td>
          <td>90.2</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s3</th>
          <td>167.6</td>
          <td>76.8</td>
          <td>M</td>
        </tr>
        <tr>
          <th>s4</th>
          <td>156.2</td>
          <td>80.4</td>
          <td>M</td>
        </tr>
        <tr>
          <th>s5</th>
          <td>161.4</td>
          <td>78.9</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s6</th>
          <td>165.4</td>
          <td>82.7</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s7</th>
          <td>126.256</td>
          <td>54.0028</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s8</th>
          <td>178.567</td>
          <td>79.1122</td>
          <td>M</td>
        </tr>
        <tr>
          <th>s9</th>
          <td>198.826</td>
          <td>88.8364</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s10</th>
          <td>163.689</td>
          <td>71.9708</td>
          <td>F</td>
        </tr>
        <tr>
          <th>s11</th>
          <td>194.533</td>
          <td>86.7758</td>
          <td>M</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    s = pd.Series([89.2, 76.4, 98.2, 75.9], index=list('abcd'))
    
    'b' in s




.. parsed-literal::

    True



