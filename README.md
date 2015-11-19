keyword_extraction_kaggle
=========================

Kaggle competition - https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction

Goal: To predict the tags on a particular Stack Exchange post given only the question text and the title of the post.

Pre-Processing: Removing Stop Words, Removing duplicates, Removing tags,Removing Code and output fragments (working only with text)

Approach1:
One vs Rest Classifiers
Term frequency – inverse document frequency used to convert the text to a vectorized representation of the text.
Lot of noise in the data in terms of the unique words being used which are not stop words.
Use different classifiers to predict 5 tags per post for different sizes of data.
Classifiers Used: Naïve Bayes, Linear SVC, SGD, Passive Aggressive Classifier

Passive Aggressive Classifier works best based on F1-score(0.393). F1-Score also increases if we use more data.

Topic Modeling used for Dimensionality Reduction. 

Approach2:
Find rules that assign tags to certain words in the Post title and body.
Probabilistic model by counting number of words, and co-occurrences of words and tags.
A is a word, and B is a tag, the rule contains two features:
 Probability: Prob(B|A) = number of times A and B occur together/ number of time A occurs
 Occurence: number of times A occurs
Simple classification rules: Based on thresholds, Pick out most probable tags

Approach 2 outperforms Approach1. F1-score = 0.691




