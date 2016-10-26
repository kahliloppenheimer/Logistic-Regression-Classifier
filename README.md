# Maximum Entropy (Multinomial Logistic Regression) Classifier
This is my implementation of a maximum entropy (logistic regression) classifier, trained on the yelp dataset to classify sentiment of text. On average test runs, this implementation takes ~43s on my macbook to extract, regularlize, train and evaluate on 15,000 randomly selected samples from the Yelp dataset. In its current configuration, classifier's accuracy averages between 72-74% on the yelp dataset. My classifier is configured to split the data into 70% training, 10% dev, and 20% evaluation.

# How to run
Look through test_maxent for examples on how to train/classify. Here are the key functions to look at.

```Python
rc = ReviewCorpus('yelp_dataset.json') # Load, regularize, and featurize documents from corpus
classifier = MaxEnt() # Initialize classifier
classifier.train(rc) # Train on review corpus
classifier.classify(rc[0]) # Classify sample document
```

# Classifier details

## Yelp Review Features
For features, I experimented with Bernouli and multinomial unigrams and bigrams. For a combination of performance and accuracy reasons, I ended up settling on Bernouli unigrams for my final feature set.

The number of features varies each run because the size of the vocabulary varies depends upon which reviews are randomly selected. However, on average I extracted 14,500 features. To achieve this few features, I aggressively regularized the dataset. I lowercased the entire corpus, removed punctuation, split on whitespace, and only took the first four characters of each token. While four characters seems drastic, it gave me my best combined accuracy (72-74%), run time (40-45s), and memory usage (peak 1.7GB).

I stored features in numpy arrays. While this uses more memory, it lets me express much of the arithmetic as efficient linear algebra operators, which I found helped the clarity of my code. Additionally, I implemented this classifier with the assumption that the features for a particular instance do not vary depending on the predicted class label. Instead, I calculate the same features for each instance, but associate a different set of weights for each possible label. I understand that this limits the generalizability of our feature functions, but I found it to be an acceptably practical solution.

## Mini-batch descent configuration
I decided to do a mini-batch gradient descent with a fixed number of iterations over the entire training set. I found more-than-acceptable performance in both accuracy and timing by performing a fixed number of iterations. It avoids the complication of dealing with detecting convergence, and increases performance because I do not need to calculate accuracy or negative log-likelihood over the dev-set after each mini-batch. Instead, I print out the negative log-likelihood after each full iteration over the dataset. In experimentation, I found that 10 iterations over the entire dataset works best.

I found that a batch size of 30 and a learning rate of .001 yielded the best performance in both accuracy and timing. I wanted to play around with increasing the batch size and decreasing the learning rate in later iterations, but I have not yet. Additionally, I used the results we computed for problem set 2 to write unit tests for my probability and gradient calculations.
