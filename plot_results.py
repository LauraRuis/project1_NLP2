import csv
import os
import shutil

import matplotlib.pyplot as plt


def plot_and_save(model, second_measure='Perplexity'):

    red = '#e41a1c'
    green = '#377eb8'
    blue = '#4daf4a'

    colours = [
        green,
        red,
        blue,
    ]

    epochs = []
    aer = []
    perplexities = []

    with open('results/{}.csv'.format(model), 'r') as file:

        reader = csv.reader(file)

        for i, line in enumerate(reader):
            if i == 0: continue
            epochs.append(int(line[0]))
            aer.append(round(float(line[1]), 2))
            perplexities.append(float(line[3]))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Epochs')

    ax1.plot(epochs, aer, '{}'.format(colours[0]))
    ax1.set_ylabel('AER (val)', color=colours[0])
    ax1.tick_params('y', colors=colours[0])

    ax2 = ax1.twinx()
    ax2.plot(epochs, perplexities, '{}'.format(colours[1]))
    ax2.set_ylabel('{} (train)'.format(second_measure), color=colours[1])
    ax2.tick_params('y', colors=colours[1])

    fig.tight_layout()
    plt.savefig('plots/{}_{}.pdf'.format(model, second_measure))


shutil.rmtree('plots/')
os.mkdir('plots/')

for model in os.listdir('results/'):
    plot_and_save(os.path.splitext(model)[0], second_measure='Perplexity')
    plot_and_save(os.path.splitext(model)[0], second_measure='ELBO')