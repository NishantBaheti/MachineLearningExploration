Hypothesis Testing
==================

Null Hypothesis (H0)
~~~~~~~~~~~~~~~~~~~~

-  To check whether claim is applicable or not
-  States that there is no significant difference between a set of a
   variable

Alternate Hypothesis (H1 / Ha)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  to challenge currently accepted state of knowledge

-  More precisely states that there is a significant difference between
   a set of a variable

   ::

        Null Hypothesis and Alternate Hypothesis are mutually exclusive

Steps for Hypothesis Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

                   Start
                    |
                    v
                   State both Hypothesis
                    |
                    v
                   Formulate analysis plan 
                    |
                    v
                   Analyze data (mean, population, z=score)
                    |
                    v
                   Interpret Results (apply decision rules)
                    |
                    V
                   End 

-  Define Hypothesis H0,Ha
-  Select test statistics whose probability distribution function can be
   found under the Null Hypothesis
-  Collect data
-  Compute test statistics and calculate p-value under null hypothesis
-  Reject Null Hyppthsis if p-value is lower then predetermined
   significance value

Types of tests
~~~~~~~~~~~~~~

-  One tailed : Region of rejection is only on one side of sampling
   distribution
-  Two tailed : Region of Rejection is on both sides of sampling
   distribution

Decision rules
~~~~~~~~~~~~~~

-  p-value
-  region of acceptance

Decision Errors
~~~~~~~~~~~~~~~

+-----------------------------------+-----------------------------------+
| Type 1 Error                      | Type 2 Error                      |
+===================================+===================================+
| Occurs when a researcher rejects  | Fails to reject an Hypothesis     |
| a Null Hypothesis when it is true | when it is false                  |
+-----------------------------------+-----------------------------------+
| False Positive                    | False Negative                    |
+-----------------------------------+-----------------------------------+
| Significance Level : probability  | Power Of Test : probability of    |
| of commiting a Type 1 error       | not commiting a Type 2 error      |
+-----------------------------------+-----------------------------------+
| alpha                             | beta                              |
+-----------------------------------+-----------------------------------+

Test Statistics
---------------

The methods used for performing t-test are shown below.

-  stats.ttest_1samp: Tests if the mean of a population is a given
   value.
-  stats.ttest_ind: Tests if the means of two independent samples are
   equal.
-  stats.ttest_rel: Tests if the means of two paired samples are equal.

.. code:: ipython3

    import numpy as np
    from scipy import stats

Example
~~~~~~~

-  Normally estimation for mean and variance of sample is made and test
   statistics is calculated
-  if population variance is identified, it is reasonable to consider
   that test statisics is normally distributed
-  if variance is unknown, sample variance is used and test statistics
   follows t distribution

.. code:: ipython3

    np.random.seed(10)
    mu, sigma = 0.8, 0.5
    significanceLevel = 0.05 ## 5%
    
    H0 = '''H0 : mean of the population is 1.0'''
    
    normDist = stats.norm(mu,sigma)
    
    ## Derive a sample 
    n = 100
    sample = normDist.rvs(n)
    
    ## compute test statistics 
    t, p = stats.ttest_1samp(sample , 1.0)
    
    print(H0)
    print(f"""
    p-value : {p}
    t-score : {t}
    """)
    
    if p < significanceLevel:
        print("H0 is rejected.")
    else:
        print("H0 is accepted.")


.. parsed-literal::

    H0 : mean of the population is 1.0
    
    p-value : 0.0013513182796454544
    t-score : -3.2984836759256875
    
    H0 is rejected.


.. code:: ipython3

    mu1, sigma1 = 0.25, 1.0
    mu2, sigma2 = 0.50, 1.0
    
    significanceLevel = 0.05 ## 5%
    
    H0 = '''H0 : population means of two random variables are equal'''
    
    normDist1 = stats.norm(mu1,sigma1)
    normDist2 = stats.norm(mu2,sigma2)
    
    ## Sample
    n = 100
    sample1 = normDist1.rvs(n)
    sample2 = normDist2.rvs(n)
    
    ## compute test statistics 
    t, p = stats.ttest_ind(sample1,sample2)
    
    print(H0)
    print(f"""
    p-value : {p}
    t-score : {t}
    """)
    
    if p < significanceLevel:
        print("H0 is rejected.")
    else:
        print("H0 is accepted.")


.. parsed-literal::

    H0 : population means of two random variables are equal
    
    p-value : 0.24665844967219017
    t-score : -1.1619402232350682
    
    H0 is accepted.


.. code:: ipython3

    mu1, sigma1 = 0.25, 1.0
    mu2, sigma2 = 0.50, 1.0
    
    significanceLevel = 0.05 ## 5%
    
    H0 = '''H0 : population means of two paired samples are equal'''
    
    normDist1 = stats.norm(mu1,sigma1)
    normDist2 = stats.norm(mu2,sigma2)
    
    ## Sample
    n = 100
    sample1 = normDist1.rvs(n)
    sample2 = normDist2.rvs(n)
    
    ## compute test statistics 
    t, p = stats.ttest_rel(sample1,sample2)
    
    print(H0)
    print(f"""
    p-value : {p}
    t-score : {t}
    """)
    
    if p < significanceLevel:
        print("H0 is rejected.")
    else:
        print("H0 is accepted.")


.. parsed-literal::

    H0 : population means of two paired samples are equal
    
    p-value : 0.3444019855090813
    t-score : -0.950046874746579
    
    H0 is accepted.

