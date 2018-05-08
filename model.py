import math
import os
import pickle
from abc import abstractmethod
from collections import Counter, defaultdict
import numpy as np
from scipy.special import digamma, loggamma

from aer import AERSufficientStatistics
from data import ParallelData, Sentence
import collections

Alignment = collections.namedtuple('Alignment', 'id word_alignments')
WordAlignment = collections.namedtuple('WordAlignment', 'sentence_number english french probability')
Metrics = collections.namedtuple('Metrics', 'loglikelihood aer perplexity')


class Model:

    use_q = None

    def __init__(self, data: ParallelData, validation_gold_alignments, initialisation_type='uniform'):

        self.data = data
        self.validation_gold_alignments = validation_gold_alignments
        self.initialisation_type = initialisation_type
        self.epochs = 0
        self.delta_sum_cache = defaultdict(lambda: {})

        self.t = self.initial_t()
        self.q_cache = self.initial_q(self.initialisation_type)

    @abstractmethod
    def initial_t(self) -> {}:
        raise NotImplementedError

    @abstractmethod
    def initial_q(self, initialisation_type):
        raise NotImplementedError

    def get_q_cache_indexes(self, i, j, l, m):
        return ((j, l, m), i)

    def em(self):

        self.epochs += 1

        word_counts, alignment_counts = self.e_step()
        self.m_step(word_counts, alignment_counts)

    def m_step(self, word_counts: {}, alignment_counts: {}):
        self.update_t(word_counts)
        self.update_q(alignment_counts)

    def e_step(self) -> ({}, {}):
        word_counts = defaultdict(lambda: Counter())
        alignment_counts = defaultdict(lambda: Counter())

        print("Training EM ...")
        for index, sentence in enumerate(self.data.train_data):

            if index % 1000 == 0:
                print('{:.0%} % of sentences processed'.format(index / len(self.data.train_data)))

            for i, english_word in enumerate(sentence.english):
                for j, french_word in enumerate(sentence.french):
                    delta = self.delta(sentence, i, j)

                    word_counts[english_word][french_word] += delta

                    first_alignment_count_index, second_alignment_count_index = self.get_q_cache_indexes(i, j, len(sentence.english), len(sentence.french))
                    alignment_counts[first_alignment_count_index][second_alignment_count_index] += delta

        for english_word in word_counts.keys():
            english_word_count = sum(word_counts[english_word].values())
            for french_word, french_word_count in word_counts[english_word].items():
                word_counts[english_word][french_word] = french_word_count / english_word_count

        return word_counts, alignment_counts

    def update_t(self, word_counts: {}):
        self.t = word_counts
        self.delta_sum_cache = defaultdict(lambda: {})  # updating t invalidates cache -- clear all values

    @abstractmethod
    def update_q(self, alignment_counts: {}):
        raise NotImplementedError

    def delta(self, sentence: Sentence, i, j):

        cached_delta_sums = self.delta_sum_cache.get(sentence.id)
        if cached_delta_sums:
            if sentence.french[j] in cached_delta_sums.keys():
                total = cached_delta_sums[sentence.french[j]]
            else:
                total = 0
                for i_prime, word in enumerate(sentence.english):
                    total += self.q(sentence, i_prime, j) * self.t[sentence.english[i_prime]][sentence.french[j]]
                self.delta_sum_cache[sentence.id][sentence.french[j]] = total
        else:
            total = 0
            for i_prime, word in enumerate(sentence.english):
                total += self.q(sentence, i_prime, j) * self.t[sentence.english[i_prime]][sentence.french[j]]

            self.delta_sum_cache[sentence.id] = {}
            self.delta_sum_cache[sentence.id][sentence.french[j]] = total

        return (self.q(sentence, i, j) * self.t[sentence.english[i]][sentence.french[j]]) \
               / total

    def get_best_alignment(self, sentence: Sentence, show_alignment=False) -> Alignment:

        l_english = len(sentence.english)
        m_french = len(sentence.french)
        best_alignment = {}

        for j in range(m_french):
            max_probability = 0
            for i in range(l_english):
                english_word = sentence.english[i]
                french_word = sentence.french[j]
                t_values = self.t.get(english_word, 0)
                if t_values != 0:
                    t_value = t_values.get(french_word, 0)
                    probability = self.q(sentence, i, j) * t_value
                else:
                    probability = 0
                if probability > max_probability:
                    max_probability = probability
                    best_alignment[j] = (i, probability)

        if len(best_alignment) > 0:
            alignment = self.transform_alignment(sentence, best_alignment)
            if show_alignment:
                self.print_alignment(sentence, alignment)
            return alignment
        else:
            return self.null_alignment(sentence)

    def get_validation_metrics(self) -> Metrics:

        log_data_probability = 0
        entropy = 0
        predicted_alignments = []

        for sentence in self.data.validation_data:

            log_sentence_probability = 0
            sentence_alignment = []

            alignment = self.get_best_alignment(sentence, False)

            for word_alignment in alignment.word_alignments:
                log_sentence_probability += math.log(word_alignment.probability)

                if word_alignment.english is not 0:
                    sentence_alignment.append(
                        (word_alignment.english, word_alignment.french + 1)  # french alignments start from 1
                    )
            entropy += - log_sentence_probability
            log_data_probability += log_sentence_probability
            predicted_alignments.append(set(sentence_alignment))

        data_probability = math.exp(log_data_probability)
        aer = AERSufficientStatistics(self.validation_gold_alignments, predicted_alignments).aer()
        perplexity = entropy

        return Metrics(data_probability, aer, perplexity)

    def write_NAACL_alignments(self, write_to, data_set="train_data"):

        print("Writing alignments ...")

        if write_to is None:
            write_to = "alignments/" + data_set + ".naacl"

        perplexity = 0
        with open(write_to, "w") as file:

            current_dataset = getattr(self.data, data_set)
            for i, sentence in enumerate(current_dataset):

                sentence_likelihood = 0
                if i % 10000 == 0:
                    print('{:.0%} of sentences processed for writing alignments'.format(sentence.id / len(current_dataset)))

                alignment = self.get_best_alignment(sentence, show_alignment=False)
                for word_alignment in alignment.word_alignments:
                    file.write(self.NAACL_word_alignment(word_alignment))
                    sentence_likelihood += math.log(word_alignment.probability)

                perplexity += -1 * sentence_likelihood
        return perplexity

    @staticmethod
    def transform_alignment(sentence: Sentence, alignment: dict) -> Alignment:
        alignment_output = []
        for french_index, (english_index, probability) in alignment.items():
            alignment_output.append(WordAlignment(sentence.id, english_index, french_index, probability))
        return Alignment(sentence.id, alignment_output)

    @staticmethod
    def null_alignment(sentence: Sentence) -> Alignment:

        null_alignment = []
        for french_id in range(len(sentence.french)):
            null_alignment.append(WordAlignment(sentence.id, 0, french_id, 1))

        return Alignment(sentence.id, null_alignment)

    @staticmethod
    def NAACL_word_alignment(alignment: WordAlignment) -> str:

        if alignment.english is not 0:
            return '{} {} {} {}\n'.format(
                alignment.sentence_number + 1,
                alignment.english,
                alignment.french + 1,
                alignment.probability
            )
        else:
            return ''

    @staticmethod
    def print_alignment(sentence: Sentence, alignment: Alignment):

        alignment_words = []
        for _, english_index, french_index, probability in alignment.word_alignments:
            alignment_words.append(sentence.english[english_index] + "- " + str(probability) + " ->" + sentence.french[french_index])

        return print(" ".join(alignment_words))

    @abstractmethod
    def q(self, sentence: Sentence, i, j):
        raise NotImplementedError

    @abstractmethod
    def path_for_parameters(self):
        raise NotImplementedError

    def save_parameters(self, directory):
        with open(os.path.join(directory, self.path_for_parameters(), 't.pickle'), 'wb') as file:
            pickle.dump(dict(self.t), file)
        with open(os.path.join(directory, self.path_for_parameters(), 'q.pickle'), 'wb') as file:
            pickle.dump(dict(self.q_cache), file)
        with open(os.path.join(directory, self.path_for_parameters(), 'epochs.pickle'), 'wb') as file:
            pickle.dump(self.epochs, file)

    def load_parameters(self, directory):
        with open(os.path.join(directory, self.path_for_parameters(), 't.pickle'), 'rb') as infile:
            self.t = pickle.load(infile)

        with open(os.path.join(directory, self.path_for_parameters(), 'q.pickle'), 'rb') as infile:
            self.q_cache = pickle.load(infile)

        with open(os.path.join(directory, self.path_for_parameters(), 'epochs.pickle'), 'rb') as file:
            self.epochs = pickle.load(file)


class Model1(Model):

    use_q = False

    def initial_t(self):

        if not self.initialisation_type == 'uniform': raise ValueError('invalid initialisation type for model 1')

        french_vocabulary_size = len(self.data.training_vocabulary_french)
        return defaultdict(lambda: defaultdict(lambda: 1 / french_vocabulary_size))

    def initial_q(self, initialisation_type):

        return {}

    def q(self, sentence: Sentence, i, j):

        # Values for q cancel out in the posterior -- exact value does not matter
        return 1

    def update_q(self, alignment_counts: {}):

        # Model 1 does not have q values
        pass

    def path_for_parameters(self):
        return "model1/"


class Model2(Model):

    use_q = True

    def initial_t(self):

        if self.initialisation_type == "uniform":
            french_vocabulary_size = len(self.data.training_vocabulary_french)
            return defaultdict(lambda: defaultdict(lambda: 1 / french_vocabulary_size))

        elif self.initialisation_type == "ibm1":
            model1 = Model1(self.data, self.validation_gold_alignments)
            model1_path = model1.path_for_parameters()
            t_model1_parameters_pickle = "parameters/" + model1_path + "t.pickle"
            if not os.path.isfile(t_model1_parameters_pickle):
                raise ValueError("Cannot initialize with Model 1 parameters, parameters-file not found")
            model1.load_parameters('parameters')

            return model1.t

        elif self.initialisation_type == 'random':

            french_words = self.data.training_vocabulary_french
            french_words.add('<UNK>')

            scores = np.random.rand(len(self.data.training_vocabulary_french))
            scores = scores / sum(scores)

            translation_probabilities = dict(zip(self.data.training_vocabulary_french, scores))

            # Use same random distribution for all english words -- quickly exceeds available memory otherwise
            return defaultdict(lambda: translation_probabilities)

        else: raise ValueError('Undefined initialisation type')

    def initial_q(self, initialisation_type):

        if initialisation_type == "uniform":
            return defaultdict(lambda: defaultdict(lambda: 1 / 100))

        elif initialisation_type == "random":

            print("initializing ..")

            init_q = defaultdict(lambda: defaultdict(lambda: 1 / 100))
            for n_sent, sentence in enumerate(self.data.train_data):
                if n_sent % 1000 == 0:
                    print('{:.0%} % of sentences processed for initialization'.format(n_sent / len(self.data.train_data)))
                for i, english_word in enumerate(sentence.english):
                    for j, french_word in enumerate(sentence.french):
                        init_q[(i, len(sentence.english), len(sentence.french))][j] = 0

            for (i, l, m), french_words in init_q.items():

                dirichlet_init = np.random.dirichlet(np.ones(len(french_words)) / 3, size=1)[0]
                for index, (j, french_word) in enumerate(french_words.items()):
                    init_q[(i, l, m)][j] = dirichlet_init[index]

            print("initialized")
            return init_q

        elif initialisation_type == 'ibm1':
            # IBM 1 does not have q parameters -- use uniform initial values instead
            return self.initial_q('uniform')

        else: raise ValueError('Undefined initialisation type')

    def q(self, sentence: Sentence, i, j):

        l = len(sentence.english)
        m = len(sentence.french)

        first_q_cache_index, second_q_cache_index = self.get_q_cache_indexes(i, j, l, m)

        q_j = self.q_cache[first_q_cache_index][second_q_cache_index]
        q_jprime = sum(self.q_cache[first_q_cache_index].values())

        if q_jprime == 0:
            # avoid division by 0 error
            return 0.000000000000001

        return q_j / q_jprime

    def update_q(self, alignment_counts: {}):

        for alignment in alignment_counts.keys():
            total = sum(alignment_counts[alignment].values())
            for count in alignment_counts[alignment].keys():
                alignment_counts[alignment][count] /= total

        self.q_cache = alignment_counts

    def path_for_parameters(self):
        return "model2-{}/".format(self.initialisation_type)


class JumpingModel2(Model2):

    def jump(self, i, j, l, m, max_jump=100):

        jump = max(int(i - math.floor(j * l / m)) + max_jump, 0)

        if jump >= 2 * max_jump:
            jump = 2 * max_jump - 1

        return jump

    def get_q_cache_indexes(self, i, j, l, m):
        return ((j, l), self.jump(i, j + 1, l, m))

    def path_for_parameters(self):
        return 'jumping-model2/'

    def initial_q(self, initialisation_type):

        if not initialisation_type == 'random': raise ValueError('undefined initialisation type')

        print("initializing ..")

        init_q = defaultdict(lambda: defaultdict(lambda: 1 / 100))
        for n_sent, sentence in enumerate(self.data.train_data):
            if n_sent % 1000 == 0:
                print('{:.0%} % of sentences processed for initialization'.format(n_sent / len(self.data.train_data)))
            for i, english_word in enumerate(sentence.english):
                for j, french_word in enumerate(sentence.french):
                    jump = self.jump(i, j, len(sentence.english), len(sentence.french), 100)
                    init_q[(j, len(sentence.english))][jump] = 0

        for (j, l), jumps in init_q.items():

            dirichlet_init = np.random.dirichlet(np.ones(len(jumps)) / 3, size=1)[0]
            for index, (jump, value) in enumerate(jumps.items()):
                init_q[(j, l)][jump] = dirichlet_init[index]

        return init_q


class BayesianModel1(Model1):

    def __init__(self, data: ParallelData, validation_gold_alignments, dirichlet_par: float):
        super(BayesianModel1, self).__init__(data, validation_gold_alignments)
        self.l = self.initial_t()
        self.alpha = dirichlet_par
        self.l_sum_cache = {}

    def update_l(self):

        self.l_sum_cache = {}
        for english_word, french_words in self.t.items():
            for index, (french_word, french_count) in enumerate(self.t[english_word].items()):
                self.l[english_word][french_word] = self.alpha + french_count

    def m_step(self, word_counts: {}, alignment_counts: {}):
        self.update_t(word_counts)
        self.update_q(alignment_counts)
        self.update_l()

    def delta(self, sentence: Sentence, i, j):

        bayesian_t = self.calculate_t(sentence.french[j], sentence.english[i])

        if sentence.french[j] in self.delta_sum_cache[sentence.id].keys():
            total = self.delta_sum_cache[sentence.id][sentence.french[j]]
        else:
            total = 0
            for i_prime, word in enumerate(sentence.english):
                total += self.q(sentence, i_prime, j) * self.calculate_t(sentence.french[j], sentence.english[i_prime])
            self.delta_sum_cache[sentence.id][sentence.french[j]] = total

        if total != 0:
            return (self.q(sentence, i, j) * bayesian_t) \
                   / total
        else:
            print("WARNING: 0 encountered in Bayesian Model 1 delta calculation for sentence {}".format(sentence.id))
            return 0

    def calculate_t(self, french_word, english_word):

        if english_word in self.l_sum_cache.keys():
            bayesian_t_sum = self.l_sum_cache[english_word]
        else:
            bayesian_t_sum = sum(self.l[english_word].values())
            self.l_sum_cache[english_word] = bayesian_t_sum
        t = math.exp(digamma(self.l[english_word][french_word]) - digamma(bayesian_t_sum))

        if t != 0:
            return t
        else:
            return 0.0000000000001

    def elbo(self):

        V_e = len(self.data.training_vocabulary_english)
        kl_divergence = 0
        lambda_sum = 0
        for english_word in self.t.keys():
            for french_word in self.t[english_word].keys():
                kl_divergence += self.t[english_word][french_word] * (self.alpha - self.l[english_word][french_word])
                kl_divergence += loggamma(self.l[english_word][french_word])
                lambda_sum += self.l[english_word][french_word]
        kl_divergence -= V_e * loggamma(self.alpha)
        kl_divergence += loggamma(V_e * self.alpha)
        kl_divergence -= loggamma(lambda_sum)
        perplexity = self.write_NAACL_alignments(None)

        kl_divergence = kl_divergence.real

        return -perplexity + kl_divergence

    def path_for_parameters(self):
        return "bayesian_model1/"


class BayesianModel2(Model2):

    def __init__(self, data: ParallelData, validation_gold_alignments, dirichlet_par: float):
        super(BayesianModel2, self).__init__(data, validation_gold_alignments)
        self.l = self.initial_t()
        self.alpha = dirichlet_par
        self.l_sum_cache = {}

    def update_l(self):

        self.l_sum_cache = {}
        for english_word, french_words in self.t.items():
            for index, (french_word, french_count) in enumerate(self.t[english_word].items()):
                self.l[english_word][french_word] = self.alpha + french_count

    def m_step(self, word_counts: {}, alignment_counts: {}):
        self.update_t(word_counts)
        self.update_q(alignment_counts)
        self.update_l()

    def delta(self, sentence: Sentence, i, j):

        bayesian_t = self.calculate_t(sentence.french[j], sentence.english[i])

        if sentence.french[j] in self.delta_sum_cache[sentence.id].keys():
            total = self.delta_sum_cache[sentence.id][sentence.french[j]]
        else:
            total = 0
            for i_prime, word in enumerate(sentence.english):
                total += self.q(sentence, i_prime, j) * self.calculate_t(sentence.french[j], sentence.english[i_prime])
            self.delta_sum_cache[sentence.id][sentence.french[j]] = total

        if total != 0:
            return (self.q(sentence, i, j) * bayesian_t) \
                   / total
        else:
            print("WARNING: 0 encountered in Bayesian Model 1 delta calculation for sentence {}".format(sentence.id))
            return 0

    def calculate_t(self, french_word, english_word):

        if english_word in self.l_sum_cache.keys():
            bayesian_t_sum = self.l_sum_cache[english_word]
        else:
            bayesian_t_sum = sum(self.l[english_word].values())
            self.l_sum_cache[english_word] = bayesian_t_sum
        t = math.exp(digamma(self.l[english_word][french_word]) - digamma(bayesian_t_sum))

        if t != 0:
            return t
        else:
            return 0.0000000000001

    def elbo(self):

        V_e = len(self.data.training_vocabulary_english)
        kl_divergence = 0
        lambda_sum = 0
        for english_word in self.t.keys():
            for french_word in self.t[english_word].keys():
                kl_divergence += self.t[english_word][french_word] * (self.alpha - self.l[english_word][french_word])
                kl_divergence += loggamma(self.l[english_word][french_word])
                lambda_sum += self.l[english_word][french_word]
        kl_divergence -= V_e * loggamma(self.alpha)
        kl_divergence += loggamma(V_e * self.alpha)
        kl_divergence -= loggamma(lambda_sum)
        perplexity = self.write_NAACL_alignments(None)

        kl_divergence = kl_divergence.real

        return -perplexity + kl_divergence

    def path_for_parameters(self):
        return "bayesian_model2/"
