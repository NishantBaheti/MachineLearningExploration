Natural Language Processing
===========================

.. code:: ipython3

    import nltk 
    
    ## nltk.download('punkt')
    ## nltk.download('book')
    ## will be downloaded in the user directory
    
    text = "Hello ! My name is Bazooka. Foo is bar and bar is foo."

.. code:: ipython3

    ## sentence tokenizer
    sentences = nltk.sent_tokenize(text)
    sentences 




.. parsed-literal::

    ['Hello !', 'My name is Bazooka.', 'Foo is bar and bar is foo.']



.. code:: ipython3

    ## word tokenizer
    words = nltk.word_tokenize(text)
    
    words




.. parsed-literal::

    ['Hello',
     '!',
     'My',
     'name',
     'is',
     'Bazooka',
     '.',
     'Foo',
     'is',
     'bar',
     'and',
     'bar',
     'is',
     'foo',
     '.']



.. code:: ipython3

    wordFreq = nltk.FreqDist(words)
    print(f"""
        Word Frequencies : {wordFreq.elements}
        2 Most Common    : {wordFreq.most_common(2)}
    """)


.. parsed-literal::

    
        Word Frequencies : <bound method Counter.elements of FreqDist({'is': 3, '.': 2, 'bar': 2, 'Hello': 1, '!': 1, 'My': 1, 'name': 1, 'Bazooka': 1, 'Foo': 1, 'and': 1, ...})>
        2 Most Common    : [('is', 3), ('.', 2)]
    


.. code:: ipython3

    ## Importing Items of book
    from nltk.book import *


.. parsed-literal::

    *** Introductory Examples for the NLTK Book ***
    Loading text1, ..., text9 and sent1, ..., sent9
    Type the name of the text or sentence to view it.
    Type: 'texts()' or 'sents()' to list the materials.
    text1: Moby Dick by Herman Melville 1851
    text2: Sense and Sensibility by Jane Austen 1811
    text3: The Book of Genesis
    text4: Inaugural Address Corpus
    text5: Chat Corpus
    text6: Monty Python and the Holy Grail
    text7: Wall Street Journal
    text8: Personals Corpus
    text9: The Man Who Was Thursday by G . K . Chesterton 1908


.. code:: ipython3

    ## find 
    print(text1.findall("<tri.*r>"))


.. parsed-literal::

    triangular; triangular; triangular; triangular
    None


.. code:: ipython3

    ## word count
    print(len(text1))


.. parsed-literal::

    260819


.. code:: ipython3

    ## unique word count 
    print(len(set(text1)))


.. parsed-literal::

    19317


.. code:: ipython3

    ## transforming words 
    print(len(set([word.lower() for word in set(text1)])))


.. parsed-literal::

    17231


.. code:: ipython3

    ## word coverage 
    print(len(text1) / len(set(text1)))


.. parsed-literal::

    13.502044830977896


.. code:: ipython3

    ## filtering 
    [word for word in set(text1) if word.startswith('Sun')]




.. parsed-literal::

    ['Sunday', 'Sunset', 'Sunda']



.. code:: ipython3

    ## frequency distribution 
    fdist = nltk.FreqDist(text1)
    
    ## print([(text,fdist[text]) for text in fdist])

Common frequenct distribution methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=========================== ============================
Method                      Discription
=========================== ============================
fdist = nltk.FreqDist(text) freq. dist. object
fdist.pprint()              print
fdist[‘exmple’]             get count
fdist.freq(‘example’)       get freq
fdist.N()                   Total number of samples
fdist.keys()                keys in desc order of freq
for text in fdist           iterate
fdist.max()                 key with max freq
fdist.tabulate()            tabulate
fdist.plot()                plot of freq dist
fdist.plot(cumulative=True) cumulative plot of freq dist
fdist1 < fdist2             compare
=========================== ============================

.. code:: ipython3

    fdist.pprint()


.. parsed-literal::

    FreqDist({',': 18713, 'the': 13721, '.': 6862, 'of': 6536, 'and': 6024, 'a': 4569, 'to': 4542, ';': 4072, 'in': 3916, 'that': 2982, ...})


.. code:: ipython3

    fdist['Sunday']




.. parsed-literal::

    7



.. code:: ipython3

    fdist.freq('Sunday')		




.. parsed-literal::

    2.683853553613809e-05



.. code:: ipython3

    fdist.N()		




.. parsed-literal::

    260819



.. code:: ipython3

    ## fdist.keys()		

.. code:: ipython3

    fdist.max()		




.. parsed-literal::

    ','



.. code:: ipython3

    ## fdist.tabulate()		

.. code:: ipython3

    
    
    fdist = nltk.FreqDist(text)
    fdist.plot()		



.. image:: NLTK_files/NLTK_21_0.png




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x7fcaffdfc490>



.. code:: ipython3

    fdist.plot(cumulative=True)		



.. image:: NLTK_files/NLTK_22_0.png




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x7fcaff5f8ee0>



.. code:: ipython3

    ## compare
    ## fdist1 < fdist2

Text Corpora /Text Corpus
-------------------------

-  Gutenberg Corpus
-  Web and Chat Text
-  Brown Corpus
-  Reuters Corpus
-  Inaugural Address Corpus
-  Annotated Text Corpora
-  etc.

Gutenberg Corpus
~~~~~~~~~~~~~~~~

::

   NLTK includes a small selection of texts from the Project Gutenberg electronic text archive, which contains some 25,000 free electronic books, hosted at. 

http://www.gutenberg.org/

Brown
~~~~~

::

   The Brown Corpus was the first million-word electronic corpus of English, 

Inaugural
~~~~~~~~~

::

   US presidential speeches 

Popular Text Corpora
--------------------

::

   stopwords : Collection of stop words.
   reuters : Collection of news articles.
   cmudict : Collection of CMU Dictionary words.
   movie_reviews : Collection of Movie Reviews.
   np_chat : Collection of chat text.
   names : Collection of names associated with males and females.
   state_union : Collection of state union address.
   wordnet : Collection of all lexical entries.
   words : Collection of words in Wordlist corpus.

Text Corpus Structure
---------------------

A text corpus is organized into any of the following four structures.

::

   Isolated - Holds Individual text collections.
   Categorized - Each text collection tagged to a category.
   Overlapping - Each text collection tagged to one or more categories, and
   Temporal - Each text collection tagged to a period, date, time, etc.

.. code:: ipython3

    from nltk.corpus import genesis 
    
    genesis.fileids()




.. parsed-literal::

    ['english-kjv.txt',
     'english-web.txt',
     'finnish.txt',
     'french.txt',
     'german.txt',
     'lolcat.txt',
     'portuguese.txt',
     'swedish.txt']



.. code:: ipython3

    from prettytable import PrettyTable
    
    x = PrettyTable()
    x.field_names = ["average word length","average sentence length","fileids"]
    for fileid in genesis.fileids():
        n_chars = len(genesis.raw(fileid))
        n_words = len(genesis.words(fileid))
        n_sents = len(genesis.sents(fileid))
        
        
        x.add_row([int(n_chars/n_words), int(n_words/n_sents), fileid])
        
    print(x)


.. parsed-literal::

    +---------------------+-------------------------+-----------------+
    | average word length | average sentence length |     fileids     |
    +---------------------+-------------------------+-----------------+
    |          4          |            30           | english-kjv.txt |
    |          4          |            19           | english-web.txt |
    |          5          |            15           |   finnish.txt   |
    |          4          |            23           |    french.txt   |
    |          4          |            23           |    german.txt   |
    |          4          |            20           |    lolcat.txt   |
    |          4          |            27           |  portuguese.txt |
    |          4          |            30           |   swedish.txt   |
    +---------------------+-------------------------+-----------------+


.. code:: ipython3

    from nltk.corpus import inaugural
    int(len(inaugural.words('1789-Washington.txt')) / len(set(inaugural.words('1789-Washington.txt'))))




.. parsed-literal::

    2



Conditional Frequency
---------------------

::

   Conditional Frequency is Frequency Distribution based on conditions.

   CFD : Conditional Frequency Distribution

.. code:: ipython3

    cItems = [
        ('F','apple'), 
        ('F','apple'), 
        ('F','kiwi'), 
        ('V','cabbage'), 
        ('V','cabbage'),
        ('V','potato') 
    ]
    cfd = nltk.ConditionalFreqDist(cItems)

.. code:: ipython3

    cfd.conditions()




.. parsed-literal::

    ['F', 'V']



.. code:: ipython3

    cfd['F']




.. parsed-literal::

    FreqDist({'apple': 2, 'kiwi': 1})



.. code:: ipython3

    cfd['V']




.. parsed-literal::

    FreqDist({'cabbage': 2, 'potato': 1})



=================================== ===================================
Method                              Description
=================================== ===================================
cfdist = ConditionalFreqDist(pairs) create
cfdist.conditions()                 show conditions
cfdist[condition]                   freq distribution for the condition
cfdist[condition][sample]           freq for the given condition
cfdist.tabulate()                   tabulate
cfdist.plot()                       plot of freq dist
cfdist.plot(cumulative=True)        cumulative plot of freq dist
cfdist1 < cfdist2                   compare
=================================== ===================================



