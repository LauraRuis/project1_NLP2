import os
import pickle

from aer import AERSufficientStatistics, read_naacl_alignments
from model import Model, Model1, BayesianModel1, Model2, BayesianModel2, JumpingModel2
from data import ParallelData, Sentence, Alignment, WordAlignment, Certainty


def write_alignments(model: Model, file_name):
    print(file_name)
    model.write_NAACL_alignments(os.path.join('predictions', file_name), 'test_data')


data_pickle = 'pickles/data-t5.pickle'
with open(data_pickle, 'rb') as file:
    data = pickle.load(file)

# model = Model1(data, None)
# model.load_parameters('parameters')
# write_alignments(model, 'ibm1.mle.naacl')
# #
# model = BayesianModel1(data, None, 0.1)
# model.load_parameters('parameters')
# write_alignments(model, 'ibm1.vb.naacl')
#
# model = Model2(data, None, 'uniform')
# model.load_parameters('parameters')
# write_alignments(model, 'ibm2-uniform.mle.naacl')
#
# model = Model2(data, None, 'random')
# model.load_parameters('parameters')
# write_alignments(model, 'ibm2-random.mle.naacl')
#
# model = Model2(data, None, 'ibm1')
# model.load_parameters('parameters')
# write_alignments(model, 'ibm2-ibm1.mle.naacl')

model = BayesianModel2(data, None, 0.1)
model.load_parameters('parameters')
write_alignments(model, 'ibm2.vb.naacl')

# model = JumpingModel2(data, None, 'random')
# model.load_parameters('parameters')
# write_alignments(model, 'ibm2-jumps.mle.naacl')

testing_gold_alignment_pickle = 'pickles/testing_gold_alignments.pickle'
with open(testing_gold_alignment_pickle, 'rb') as file:
    testing_gold_alignments = pickle.load(file)

for file in os.listdir('predictions'):
    if file.endswith('.naacl'):
        predictions = []
        for prediction in read_naacl_alignments('predictions/{}'.format(file)):
            predictions.append(prediction[0])
        aer = AERSufficientStatistics(testing_gold_alignments, predictions).aer()
        print('{}: {}'.format(file, round(aer, 5)))
