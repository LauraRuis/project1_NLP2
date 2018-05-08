from typing import List, Dict

import matplotlib.pyplot as plt
from collections import defaultdict

from data import Sentence

figure_path = 'figures/'


def _plot_words(axes, xs, ys, words, vertical_position, weight='bold', color='black'):
    """

    Plots a pair of sentences at given positions, on a given axes object.

    Args:
       axes (pyplotlib axes object): axes on which to plot
       xs (list of floats): x coordinates
       ys (list of floats): y coordinates
       words (list of strings): words to be displayed
       vertical_position (string): where words should be displayed relative to point coordinates
       weight (string): font weight
       color (string or list of strings): color/s to be used for displaying words
    """
    for n in range(0, len(words)):
        axes.text(xs[n], ys[n], words[n], size=9, family='sans-serif',
                  weight=weight, color=color,
                  horizontalalignment='center',
                  verticalalignment=vertical_position)


def _get_coordinates(bitext, draw_all=False, one_sent=False, sent_index=0, word_index=0):
    """
    Generates x and y coordinates to be used for plotting sentence pairs
    and alignment links.

    Args:
       bitext (list of tuples): list of translation pairs
       one_sent (Boolean): whether coordinates ahould be generated for one
           selected sentence pair and one word in it, or for the whole bitext
       sent_index (int): index of the selected sentence pair
       word_index (int): index of the target foreign word
    """
    x_positions_f = []
    y_positions_f = []
    x_positions_e = []
    y_positions_e = []
    edge_pos = []
    words_f = []
    words_e = []
    sents, alignments = bitext
    for (n, (f, e)) in enumerate(sents):

        for j in range(0, len(f)):
            x_positions_f.append(j+1)
            y_positions_f.append((3*n)-2)
            words_f.append(f[j])
            if (not one_sent) or (one_sent and word_index==j):
                for i in range(0, len(e)):
                    if draw_all:
                        edge_pos.append([[j+1, i+1], [(3*n)-1.9, (3*n)-1.1]])
                    else:
                        if i in alignments[j]:
                            edge_pos.append([[j+1, i+1], [(3*n)-1.9, (3*n)-1.1]])

        for i in range(0, len(e)):
            x_positions_e.append(i+1)
            y_positions_e.append((3*n)-1)
            words_e.append(e[i])
    coord_dict = {'x_f': x_positions_f, 'x_e': x_positions_e,
            'y_f': y_positions_f, 'y_e': y_positions_e,
            'edges': edge_pos, 'w_f': words_f, 'w_e': words_e}
    return coord_dict


def draw_alignment_from_file(naacl_path, french_path, english_path, file_name: str, sure=False, sentence_id=1):
    """
    input:
        naacl_path, file with gold alignments
        french_path, french sentences
        english_path, enlgish sentences
        fig_path, output figure path
        sure, print sure alignments
        sentence, position id of sentence to print form the corpus

    """
    french = _read_sentences_from_file(french_path, sentence_id)
    english = _read_sentences_from_file(english_path, sentence_id)
    alignments, _ = _read_alignment_from_file(naacl_path, sentence_id, sure, weighted=False)

    draw_alignment(alignments, [], Sentence(sentence_id, english, french), file_name)


def _read_sentences_from_file(path: str, sentence_id: int = 1) -> List[str]:

    with open(path, 'r') as file:
        sentences = file.readlines()

    return sentences[sentence_id - 1].split()


def _read_alignment_from_file(path: str, sentence_id: int = 1, sure: bool = False, weighted: bool = False) -> (Dict[int, List], []):

    with open(path, 'r') as file:

        lines = [line.split() for line in file.readlines()]
        sentence = filter(lambda line: int(line[0]) == sentence_id, lines)

    alignments = defaultdict(list)
    weights = []
    for line in sentence:
        if not sure or not line[3] == 'P':
            alignments[int(line[2]) - 1].append(int(line[1]) - 1)

        if weighted:
            weights.append(float(line[3]))

    return alignments, weights


def draw_alignment(alignment, prediction_weights, sentence: Sentence, output_file: str):

    bitext = ([(sentence.french, sentence.english)], alignment)

    fig = plt.figure(figsize=(40, 4))
    ax = plt.axes()
    plt.axis('off')
    coordinates = _get_coordinates(bitext)

    if len(prediction_weights) == 0:
        prediction_weights = [0.1] * len(coordinates['edges'])
    line_weights = [w * 10 for w in prediction_weights]

    lines = [ax.plot(xy[0], xy[1], alpha=0.9, linewidth=w, linestyle='-', color='#1a75ff', solid_capstyle='round')[0]
        for xy, w in zip(coordinates['edges'], line_weights)]

    ax.scatter(coordinates['x_f'] + coordinates['x_e'], coordinates['y_f'] + coordinates['y_e'],
               s=30, c='white', marker='o', lw=0, alpha=1)

    _plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
    _plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')

    plt.savefig(figure_path + output_file)


def draw_weighted_alignment_from_file(alignment_path, french_path, english_path, output_file, sure=False, sentence_id=1):
    """
    Draws an alignment that is weighted according to probs of alignment.
    We use the last column (4th) for the probability of alignment.
    """
    french = _read_sentences_from_file(french_path, sentence_id)
    english = _read_sentences_from_file(english_path, sentence_id)
    # alignments, _ = _read_alignment_from_file(naacl_path, sentence_id, sure)
    alignments, prediction_weights = _read_alignment_from_file(alignment_path, sentence_id, sure, weighted=True)

    draw_alignment(alignments, prediction_weights, Sentence(sentence_id, english, french), output_file)


if __name__ == '__main__':

    # format gold alignment data
    # sentence_no position_L1 position_L2 [S|P]
    naacl_path = 'data/validation/dev.wa.nonullalign'
    # predicted alignments
    # sentence_no position_L1 position_L2 prob_alignment
    pred_path = 'alignments/validation_data.txt'
    # French sentences
    french_path = 'data/validation/dev.f'
    # English sentences
    english_path = 'data/validation/dev.e'

    # draws gold Probable alignments
    draw_alignment_from_file(naacl_path, french_path, english_path, 'test1.pdf', sure=False, sentence_id=2)
    # draws gold Sure alignments
    draw_alignment_from_file(naacl_path, french_path, english_path, 'test2.pdf', sure=True, sentence_id=2)
    # draws predicted alingments
    draw_weighted_alignment_from_file(pred_path, french_path, english_path, 'test3.pdf', sure=False, sentence_id=10)
