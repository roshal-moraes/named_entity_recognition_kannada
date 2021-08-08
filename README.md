#Named Entity Recognition in Indian Language(Kannada)
Named Entity Recognition is a subtask of NLP aiming to identify real-world entities in texts, such as names of persons, organizations, and locations, among others. Named Entity Recognition (NER) is an important task in Natural Language Processing (NLP) applications like Information Extraction, Question Answering etc. In this project, we have proposed a system to recognize Kannada named entities like person name, location name, organization name, number, date, community for historical text. Kannada Historical corpus from various sites have been mined to obtain required dataset.

##Training NER Model:
1. 	The first thing to do is to find out a suitable dataset of historical articles and their NER labels.
2.	After getting a suitable dataset, the next step is Data Preprocessing. In this, words are cleaned and tokenized(Split paragraphs to sentences if applicable). This step is done using standard nlp functions available in Python
3. 	POS tagging of each token is performed using Viterbi algorithm. This gives us an extra feature to train and work with. Store output in a suitable output file. 
[For Viterbi code reference, please click](https://github.com/rajesh-iiith/POS-Tagging-and-CYK-Parsing-for-Indian-Languages)
4. Add sentence number for each word, Thus, we have four features(Sentence_numberWord, Part of Speech NER Tag(Output)) Refer sentence_num.py.
5.	Features extracted from the tagged corpus.
6.	Corpus divided into train and test set
7.	After all the initial steps of data preprocessing is done, the CRF model is trained using training data
8.	The model is tested on test data.
9.	Results evaluated based on Precision, Recall and F1 Score.
10. Generate pickle file for further use
Steps 4 to 9 performed in code nertraining.py
[Reference fo code:](https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2)

##Training Results:

Evaluation Output
               Precision    Recall   f1-score   

       B-COM       0.65      0.51      0.57       
      B-DATE       0.68      0.30      0.42       
       B-LOC       0.66      0.38      0.48      
       B-NUM       0.72      0.65      0.68       
       B-ORG       0.65      0.26      0.37      
       B-PER       0.67      0.47      0.55       
       B-ROL       0.53      0.18      0.27       
       I-COM       0.27      0.07      0.12       
       I-LOC       0.37      0.14      0.20       
       I-ORG       0.61      0.47      0.53        
       I-PER       0.70      0.53      0.60      
       I-ROL       0.50      0.13      0.21       
           O       0.90      0.98      0.94     
     

    Accuracy                           0.88     
Weighted Avg.       0.86      0.88      0.86     



##Using the trained model:
1. If input, is a paragraph, split it into words and eliminate all punctuations.
2. Perform POS tagging using Viterbi as above and store in file(Each word in new line, tab separated)
3. Provide sentence number for each word.
3. Use pickle file to predict outcome of each word(Refer ner.py).
4. For presentation, stem each word and display words a list in each category(Refer output_list.py)
[For Stemming code reference, click here](https://github.com/Sahana-M/shabdkosh/blob/master/Kannada-stemmer/Final_Kannada_Stemmer.py)

