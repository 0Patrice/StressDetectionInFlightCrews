import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import pandas as pd


def plotViolin(index, valueName):
    g = sns.boxplot(x="Scenario", y=f"{valueName}", hue="SP", ax=axes[index], data=pdData,
                    palette="Set2", width=0.6, linewidth=1)

    for artist in axes[index].lines:
        artist.set_zorder(10)

    for artist in axes[index].findobj(PathCollection):
        artist.set_zorder(11)

    axes[index].set_ylim([-2, 22])

    # Legend and Ticks only on final row
    if index == 6:
        axes[index].legend(handles=axes[index].legend_.legendHandles, labels=['SPO', 'DPO', 'SPO', 'DPO'],
                           ncol=2, fancybox=True, loc='lower center', bbox_to_anchor=(0.5, -0.5))
    else:
        axes[index].legend([], [], frameon=False)

if __name__ == '__main__':
    pdData = pd.read_csv('Data/results-survey732144.csv', sep=',')

    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 12))

    plotViolin(0, "MD")
    plotViolin(1, "PhD")
    plotViolin(2, "TD")
    plotViolin(3, "Perf")
    plotViolin(4, "Effort")
    plotViolin(5, "FD")
    plotViolin(6, "Realism")

    fig.subplots_adjust(left=0.06,
                        bottom=0.08,
                        right=0.96,
                        top=0.99,
                        wspace=0.2,
                        hspace=0)

    plt.show()
