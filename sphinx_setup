Spam Data Analytics
===================

.. code:: ipython3

    import pandas as pd 
    import numpy as np 
    import csv

.. code:: ipython3

    # with open("./SMSSpamCollection.csv",'r+') as data:
    #     for line in data:
    #         print(line.rstrip())

.. code:: ipython3

    df = pd.read_csv("./SMSSpamCollection.csv",sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])
    
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
          <th>label</th>
          <th>message</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>ham</td>
          <td>Go until jurong point, crazy.. Available only ...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>ham</td>
          <td>Ok lar... Joking wif u oni...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>spam</td>
          <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>ham</td>
          <td>U dun say so early hor... U c already then say...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>ham</td>
          <td>Nah I don't think he goes to usf, he lives aro...</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df.groupby('label').describe()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead tr th {
            text-align: left;
        }
    
        .dataframe thead tr:last-of-type th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr>
          <th></th>
          <th colspan="4" halign="left">message</th>
        </tr>
        <tr>
          <th></th>
          <th>count</th>
          <th>unique</th>
          <th>top</th>
          <th>freq</th>
        </tr>
        <tr>
          <th>label</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>ham</th>
          <td>4827</td>
          <td>4518</td>
          <td>Sorry, I'll call later</td>
          <td>30</td>
        </tr>
        <tr>
          <th>spam</th>
          <td>747</td>
          <td>653</td>
          <td>Please call our customer service representativ...</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    target = df['label']

NLTK
----

.. code:: ipython3

    import nltk 
    # nltk.download('all')
    
    from nltk.tokenize import word_tokenize

Tokenization
------------

.. code:: ipython3

    def splitIntoTokens(text):
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens


.. code:: ipython3

    df['tokenized_message'] = df['message'].apply(splitIntoTokens)

Lemmatization (convert a word into its base form)
-------------------------------------------------

.. code:: ipython3

    from nltk.stem.wordnet import WordNetLemmatizer
    
    def getLemmas(tokens):
        lemmas = []
        lemmatizer = WordNetLemmatizer()
        for token in tokens:
            lemmas.append(lemmatizer.lemmatize(token))
        return lemmas
    
    df['lemmatized_message'] = df['tokenized_message'].apply(getLemmas)

.. code:: ipython3

    df.iloc[11]




.. parsed-literal::

    label                                                              spam
    message               SIX chances to win CASH! From 100 to 20,000 po...
    tokenized_message     [six, chances, to, win, cash, !, from, 100, to...
    lemmatized_message    [six, chance, to, win, cash, !, from, 100, to,...
    Name: 11, dtype: object



Removing Stop Words
-------------------

.. code:: ipython3

    from nltk.corpus import stopwords
    
    stopWords = set(stopwords.words('english'))

.. code:: ipython3

    def removeStopWords(lemmas):
        filteredSentence = []
        filteredSentence = ' '.join([word for word in lemmas if word not in stopWords])
        return filteredSentence 
    
    df['filtered_message'] = df['lemmatized_message'].apply(removeStopWords)

Bag Of Words
------------

Term Document Matrix
--------------------

-  The Term Document Matrix (TDM) is a matrix that contains the
   frequency of occurrence of terms in a collection of documents.
-  In a Term Frequency Inverse Document Frequency (TFIDF) matrix, the
   term importance is expressed by Inverse Document Frequency (IDF)
-  IDF diminishes the weight of the most commonly occurring words and
   increases the weightage of rare words.

.. code:: ipython3

    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    
    tfidfVectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df = (1/len(df['label'])), 
        max_df = 0.7
    )


.. code:: ipython3

    tfidfModel = tfidfVectorizer.fit(df['filtered_message'])

.. code:: ipython3

    xMatrix = tfidfModel.transform(df['filtered_message']).toarray()

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    
    xTrain,xTest,yTrain,yTest = train_test_split(xMatrix,df['label'],test_size=0.1,random_state=7)

Decision Tree Classification
----------------------------

.. code:: ipython3

    from sklearn.tree import DecisionTreeClassifier
    
    seed = 7
    dtClassifier = DecisionTreeClassifier(random_state=seed)
    
    dtModel = dtClassifier.fit(xTrain,yTrain)
    
    yPredictDT = dtModel.predict(xTest)
    
    dtScore = dtClassifier.score(xTest,yTest)
    
    print("Decision Tree Score :",dtScore)


.. parsed-literal::

    Decision Tree Score : 0.967741935483871


Gaussian Naive Bayes
--------------------

.. code:: ipython3

    from sklearn.naive_bayes import GaussianNB
    
    gnbClassifier = GaussianNB()
    
    gnbModel = gnbClassifier.fit(xTrain,yTrain)
    
    yPredictGNB = gnbModel.predict(xTest)
    
    gnbScore = gnbModel.score(xTest,yTest)
    
    print("Gaussian Naive Bayes Score :",gnbScore)


.. parsed-literal::

    Gaussian Naive Bayes Score : 0.9086021505376344


Stochastic Gradient Descent
---------------------------

.. code:: ipython3

    from sklearn.linear_model import SGDClassifier
    
    sgdClassifier = SGDClassifier(loss='modified_huber', shuffle=True,random_state=seed)
    
    sgdModel = sgdClassifier.fit(xTrain,yTrain)
    
    yPredictSGD = sgdModel.score(xTest,yTest)
    
    sgdScore = sgdModel.score(xTest,yTest)
    
    print("Stochastic Gradient Descent Classification score :",sgdScore)


.. parsed-literal::

    Stochastic Gradient Descent Classification score : 0.9713261648745519


Support Vector Machine
----------------------

.. code:: ipython3

    from sklearn.svm import SVC
    
    svClassifier = SVC(kernel="linear", C=0.025,random_state=seed)
    
    svModel = svClassifier.fit(xTrain, yTrain)
    
    yPredictSV = svClassifier.predict(xTest)
    
    svScore = svClassifier.score(xTest, yTest)
    
    print('SVM Classifier : ',svScore)


.. parsed-literal::

    SVM Classifier :  0.8566308243727598


Random Forest
-------------

.. code:: ipython3

    from sklearn.ensemble import RandomForestClassifier
    
    rfClassifier = RandomForestClassifier(max_depth=5, n_estimators=15, max_features=60,random_state=seed)
    
    rfModel = rfClassifier.fit(xTrain, yTrain)
    
    yPredictRF = rfClassifier.predict(xTest)
    
    rfScore = rfClassifier.score(xTest, yTest)
    
    print('Random Forest Classifier : ',rfScore)


.. parsed-literal::

    Random Forest Classifier :  0.8566308243727598


.. code:: ipython3

    seed=7
    from sklearn.model_selection import StratifiedShuffleSplit
    ###cross validation with 10% sample size
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.1, random_state=seed)
    sss.get_n_splits(xMatrix,df['label'])
    print(sss)


.. parsed-literal::

    StratifiedShuffleSplit(n_splits=1, random_state=7, test_size=0.1,
                train_size=None)


.. code:: ipython3

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC,LinearSVC
    from sklearn.multiclass import OneVsRestClassifier
    
    
    classifiers = [
        DecisionTreeClassifier(),
        GaussianNB(),
        SGDClassifier(loss='modified_huber', shuffle=True),
        SVC(kernel="linear", C=0.025),
        KNeighborsClassifier(),
        OneVsRestClassifier(LinearSVC()),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10),
        AdaBoostClassifier(),
       ]
    for clf in classifiers:
        score=0
        for train_index, test_index in sss.split(xMatrix,df['label']):
            X_train, X_test = xMatrix[train_index], xMatrix[test_index]
            y_train, y_test = df['label'][train_index], df['label'][test_index]
            clf.fit(X_train, y_train)
            score=score+clf.score(X_test, y_test)
        print(clf,score)


::


    ---------------------------------------------------------------------------

    MemoryError                               Traceback (most recent call last)

    <ipython-input-24-5c554a0a5417> in <module>
         23         X_train, X_test = xMatrix[train_index], xMatrix[test_index]
         24         y_train, y_test = df['label'][train_index], df['label'][test_index]
    ---> 25         clf.fit(X_train, y_train)
         26         score=score+clf.score(X_test, y_test)
         27     print(clf,score)


    ~/anaconda3/lib/python3.8/site-packages/sklearn/tree/_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        888         """
        889 
    --> 890         super().fit(
        891             X, y,
        892             sample_weight=sample_weight,


    ~/anaconda3/lib/python3.8/site-packages/sklearn/tree/_classes.py in fit(self, X, y, sample_weight, check_input, X_idx_sorted)
        154             check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
        155             check_y_params = dict(ensure_2d=False, dtype=None)
    --> 156             X, y = self._validate_data(X, y,
        157                                        validate_separately=(check_X_params,
        158                                                             check_y_params))


    ~/anaconda3/lib/python3.8/site-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
        427                 # :(
        428                 check_X_params, check_y_params = validate_separately
    --> 429                 X = check_array(X, **check_X_params)
        430                 y = check_array(y, **check_y_params)
        431             else:


    ~/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py in inner_f(*args, **kwargs)
         71                           FutureWarning)
         72         kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
    ---> 73         return f(**kwargs)
         74     return inner_f
         75 


    ~/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)
        597                     array = array.astype(dtype, casting="unsafe", copy=False)
        598                 else:
    --> 599                     array = np.asarray(array, order=order, dtype=dtype)
        600             except ComplexWarning:
        601                 raise ValueError("Complex data not supported\n"


    ~/anaconda3/lib/python3.8/site-packages/numpy/core/_asarray.py in asarray(a, dtype, order)
         83 
         84     """
    ---> 85     return array(a, dtype, copy=False, order=order)
         86 
         87 


    MemoryError: Unable to allocate 769. MiB for an array with shape (5016, 40205) and data type float32

