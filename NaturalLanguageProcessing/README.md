## Contents 

### NLTK library

1. tokenize
1. FreqDist
1. findall
1. pprint
1. freq
1. plot 
1. Text Corpora / Corpus 
1. pretty table 

### Common frequenct distribution methods 

| Method | Discription |
| ------ | ----------- | 
| fdist = nltk.FreqDist(text) | freq. dist. object | 
| fdist.pprint() | print |
| fdist['exmple'] | get count |
| fdist.freq('example') | get freq |
| fdist.N() | Total number of samples |
| fdist.keys() | keys in desc order of freq |
| for text in fdist | iterate |
| fdist.max() | key with max freq |
| fdist.tabulate() | tabulate |
| fdist.plot() | plot of freq dist |
| fdist.plot(cumulative=True) | cumulative plot of freq dist |
| fdist1 < fdist2 | compare |

