import collections
import os
from enum import Enum
from typing import List

import pickle

Sentence = collections.namedtuple('Sentence', 'id, english, french')
Alignment = collections.namedtuple('Alignment', 'id word_alignments')
WordAlignment = collections.namedtuple('WordAlignment', 'certainty english french')


class Certainty(Enum):
    P = 'P'
    S = 'S'


class ParallelData(object):
    """
    Read in parallel dataset from HLT-NAACL
    http://www.cs.unt.edu/~rada/wpt
    """

    def __init__(self, path: str, threshold: int):

        self.english_word_counts = collections.Counter()
        self.french_word_counts = collections.Counter()

        self.training_vocabulary_english = set()
        self.training_vocabulary_french = set()

        # file names
        train_file_e = path + "training/hansards.36.2.e"
        train_file_f = path + "training/hansards.36.2.f"
        validation_file_e = path + "validation/dev.e"
        validation_file_f = path + "validation/dev.f"
        test_file_e = path + "testing/test/test.e"
        test_file_f = path + "testing/test/test.f"
        validation_file_alignments = path + "validation/dev.wa.nonullalign"
        test_file_alignments = path + "testing/answers/test.wa.nonullalign"

        # read data
        self.train_data = self.read_sentence_pairs(train_file_e, train_file_f)
        self.validation_data = self.read_sentence_pairs(validation_file_e, validation_file_f)
        self.test_data = self.read_sentence_pairs(test_file_e, test_file_f)

        # read alignments
        self.validation_data_alignments = self.read_alignments(validation_file_alignments)
        self.test_data_alignments = self.read_alignments(test_file_alignments)

        # replace unknown words
        self.replace_unknown(threshold)

    def read_alignments(self, file: str) -> []:

        alignments = []
        raw_alignments = self.read_file(file)
        current_sentence_index = 0
        current_sentence = []

        for alignment in raw_alignments:

            alignment = alignment.split()
            if len(alignment) == 0:
                continue

            # datafiles numbers from 1 instead of 0
            sentence_number = int(alignment[0]) - 1
            index_e = int(alignment[1]) - 1
            index_f = int(alignment[2]) - 1

            # sentence is finished, append to alignments and init new sentence
            if sentence_number != current_sentence_index:

                alignments.append(Alignment(
                    current_sentence_index, current_sentence
                ))
                current_sentence_index += 1
                current_sentence = []

            current_sentence.append(
                WordAlignment(
                    Certainty(alignment[3]),
                    index_e,
                    index_f
                )
            )

        return alignments

    def read_sentence_pairs(self, file_e: str, file_f: str) -> (List[Sentence], int):

        data = []
        e_sentences = self.read_file(file_e)
        f_sentences = self.read_file(file_f)
        next_id = 0

        for english, french in zip(e_sentences, f_sentences):
            french_sentence = french.split()
            sentence = Sentence(
                next_id,
                ['<NULL>'] + english.split(),
                french_sentence
            )
            data.append(sentence)

            for word in sentence.french:
                self.french_word_counts[word] += 1

            for word in sentence.english:
                self.english_word_counts[word] += 1

            next_id += 1
        return data

    def replace_unknown(self, threshold: int = 1):

        for sentence in self.train_data:
            for index, word in enumerate(sentence.french):
                if self.french_word_counts[word] <= threshold:
                    sentence.french[index] = '<UNK>'
                else:
                    self.training_vocabulary_french.add(word)

            for index, word in enumerate(sentence.english):
                if self.english_word_counts[word] <= threshold:
                    sentence.english[index] = '<UNK>'
                else:
                    self.training_vocabulary_english.add(word)

        for sentence in self.validation_data:
            for index, word in enumerate(sentence.french):
                if not word in self.training_vocabulary_french:
                    sentence.french[index] = '<UNK>'
            for index, word in enumerate(sentence.english):
                if not word in self.training_vocabulary_english:
                    sentence.english[index] = '<UNK>'

        for sentence in self.test_data:
            for index, word in enumerate(sentence.french):
                if not word in self.training_vocabulary_french:
                    sentence.french[index] = '<UNK>'
            for index, word in enumerate(sentence.english):
                if not word in self.training_vocabulary_english:
                    sentence.english[index] = '<UNK>'


    @staticmethod
    def read_file(filename: str) -> []:

        with open(filename) as infile:
            return infile.readlines()


if __name__ == '__main__':

    threshold = 5
    filename = 'pickles/data-t{}.pickle'.format(threshold)

    if not os.path.isfile(filename):

        data = ParallelData('data/', threshold)

        with open(filename, 'wb') as file:
            pickle.dump(data, file)