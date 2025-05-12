# WordNet_in_DM
This repository is created for the work [Putting WordNet’s Dictionary Examples in the Context of Definition Modelling: An Empirical Analysis](https://aclanthology.org/2022.cogalex-1.6/) that is submitted to CogALex 2022, and its extention [WordNet under Scrutiny: Dictionary Examples in the Era of Large Language Models](https://aclanthology.org/2024.lrec-main.1538/)</br>

## Datasets

[**WordNet**](datasets/WordNet): from NLTK, we extracted all WordNet lemmas that have examples, and then we divided them randomly (80% for training (90% training and 10% validation), and 20% for testing). </br>

[**CHA**](datasets/CHA): this dataset based on Oxford Dictionaries. Its original splits are available here and also we created random splits from them that have the same size as WordNet files to be used in our experiment.</br>

## Source codes
[**dm_training_testing.py**](src/dm_training_testing.py): to train a definition generation model using BART as Seq2Seq model. It takes 3 input files (training, validation, and testing files) and the prompt type or encoding method which are ['lemma:' or 'target']. 'lemma:' means without using any special tokens to identify the target lemma in the context while 'target' uses special tokens \<target> and \</target> around the target lemma. 

```
python3 src/dm_training_testing.py -train PATH_TO_TRAINING_FILE -val PATH_TO_VALIDATION_FILE -test PATH_TO_TESTING_FILE -o PATH_TO_OUTPUT_FILE -p PROMPT_TYPE
```

[**Intrinsic evaluation**](https://colab.research.google.com/drive/18kXRLXlEm-2uku5Imw0jzttqw5O2n7c6?usp=sharing): this Google colab notebook evaluates the definition model interinisically using BLEU, METEOR, ROUGE, and BERTScore. It takes the output file from [dm_training_testing.py](src/dm_training_testing.py) to evaluate the generated definitions.</br>


[**Calculate GDEX**](https://colab.research.google.com/drive/1qK8wriSzi6gGxjwYa3tHjpXJhsES9QIE?usp=sharing): this Google colab notebook calculates the scores of GDEX factors for given dataset examples. Pronouns and frequent words lists that are used in this notebook are available in [GDEX_files](datasets/GDEX_files).



