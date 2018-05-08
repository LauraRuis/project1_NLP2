# An Evaluation of IBM Models 1 and 2

This project gives an implementation of the IBM models 1 and 2 for word alignments in machine translation, originally introduced by [Brown et al. (1993)](http://www.aclweb.org/anthology/J93-2003).

Refinements of the original models with variational inference [Mermer and Saraclar, 2011](http://delivery.acm.org/10.1145/2010000/2002775/p182-mermer.pdf?ip=185.56.227.5&id=2002775&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1525776727_12adb68ad5cf09e7b29f1ce312433a15) and a cheaper reparameterisation following [Vogel et al (1996)](http://www.aclweb.org/anthology/C96-2141) are also given.

All models can be found in [model.py](model.py).

## Run the models

Make sure to add training data to the [data directory](data/).
All experiments were carried out on the Hansard corpus.

There are scripts provided for model training ([train_model_1.py](train_model_1.py), [train_model_2.py](train_model_2.py), [train_bayesian_model_1.py](train_bayesian_model_1.py), [train_bayesian_model_2.py](train_bayesian_model_2.py)) that run for 10 epochs each.
Provide one of 'uniform', 'random', 'ibm1' as a parameter indicating the initialisation type.

Model parameters will be saved to the respective subdirectory in [parameters/](parameters/) along with a csv file detailing evaluation metrics after each epoch.

Use the [predict.py](predict.py) script to get predictions on the test set from the trained models.
