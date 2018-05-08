import csv
import os
import pickle

import sys

from aer import read_naacl_alignments
from data import ParallelData, Sentence, Alignment, WordAlignment, Certainty
from model import Model2, JumpingModel2

initialisation_type = sys.argv[1]

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
t_model2_parameters_pickle = "parameters/jumping-model2-{}/t.pickle".format(initialisation_type)
q_model2_parameters_pickle = "parameters/jumping-model2-{}/q.pickle".format(initialisation_type)

data_pickle = 'pickles/data-t5.pickle'
if os.path.isfile(data_pickle):
    with open(data_pickle, 'rb') as file:
        data = pickle.load(file)
else:
    data = ParallelData('data/', 5)
    with open(data_pickle, 'wb') as file:
        pickle.dump(data, file)


# data.train_data = data.train_data[1:50]


# initialize model and load saved parameters
epochs = 10

model = JumpingModel2(data, validation_gold_alignments, initialisation_type=initialisation_type)

# train model
if os.path.isfile(t_model2_parameters_pickle):
    print("Loading parameters")
    model.load_parameters('parameters')

likelihoods = []
aer = []
perplexities = []
best_score = 1

for i in range(epochs):
    print("Epoch {}".format(i + 1))

    # em and save pars
    model.em()

    # get metrics on validation set
    print("Getting validation metrics ...")
    metrics = model.get_validation_metrics()
    if metrics.aer < best_score:
        model.save_parameters('parameters')
        best_score = metrics.aer
        print("New best AER: {}".format(best_score))

    perplexity = model.write_NAACL_alignments()

    likelihoods.append(metrics.loglikelihood)
    aer.append(metrics.aer)
    perplexities.append(perplexity)

    # write metrics to file
    with open('parameters/jumping-model2-{}/training_measures.csv'.format(initialisation_type), 'w') as file:
        filewriter = csv.writer(file)
        filewriter.writerow(['epoch', 'aer', 'likelihood'])
        for epoch in range(len(aer)):
            filewriter.writerow([epoch, aer[epoch], likelihoods[epoch], perplexities[epoch]])
