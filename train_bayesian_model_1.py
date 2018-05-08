import csv
import os

import pickle

from aer import read_naacl_alignments
from data import ParallelData, Sentence, Alignment, WordAlignment, Certainty
from model import BayesianModel1

validation_gold_alignment_pickle = 'pickles/validation_gold_alignments.pickle'
testing_gold_alignment_pickle = 'pickles/testing_gold_alignments.pickle'

if os.path.isfile(validation_gold_alignment_pickle):
    with open(validation_gold_alignment_pickle, 'rb') as file:
        validation_gold_alignments = pickle.load(file)
else:
    validation_gold_alignments = read_naacl_alignments('data/validation/dev.wa.nonullalign')
    with open(validation_gold_alignment_pickle, 'wb') as file:
        pickle.dump(validation_gold_alignments, file)

if os.path.isfile(testing_gold_alignment_pickle):
    with open(testing_gold_alignment_pickle, 'rb') as file:
        testing_gold_alignments = pickle.load(file)
else:
    testing_gold_alignments = read_naacl_alignments('data/testing/answers/test.wa.nonullalign')
    with open(testing_gold_alignment_pickle, 'wb') as file:
        pickle.dump(testing_gold_alignments, file)

# loading data
t_bayesian_parameters_pickle = "parameters/bayesian_model1/t.pickle"
q_bayesian_parameters_pickle = "parameters/bayesian_model1/q.pickle"

data_pickle = 'pickles/data-t5.pickle'
if os.path.isfile(data_pickle):
    with open(data_pickle, 'rb') as file:
        data = pickle.load(file)
else:
    data = ParallelData('data/', 5)
    with open(data_pickle, 'wb') as file:
        pickle.dump(data, file)

# initialize model and load saved parameters
epochs = 10

model = BayesianModel1(data, validation_gold_alignments, 0.1)

# train model
if os.path.isfile(t_bayesian_parameters_pickle):
    model.load_parameters('parameters')

likelihoods = []
aer = []
elbos = []

best_aer = 1  # highest possible score for aer

for i in range(epochs):
    print("Epoch {}".format(i + 1))

    # em and save pars
    model.em()

    # get metrics on validation set
    metrics = model.get_validation_metrics()
    if metrics.aer < best_aer:
        model.save_parameters('parameters')
        best_aer = metrics.aer
        print("New best AER: {}".format(best_aer))

    # perplexity = model.write_NAACL_alignments("train_data")
    elbo = model.elbo()
    likelihoods.append(metrics.loglikelihood)
    aer.append(metrics.aer)
    elbos.append(elbo)
    # perplexities.append(perplexity)

    # write metrics to file
    with open('parameters/bayesian_model1/training_measures.csv', 'w') as file:
        filewriter = csv.writer(file)
        filewriter.writerow(['epoch', 'aer', 'likelihood', 'perplexity'])
        for epoch in range(len(aer)):
            filewriter.writerow([epoch, aer[epoch], likelihoods[epoch], elbos[epoch]])
