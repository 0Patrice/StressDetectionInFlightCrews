import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import pandas as pd

pdData = pd.read_csv('Data/results-survey732144.csv', sep=',')

fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 12))


def plotViolin(index, valueName):
    g = sns.violinplot(x="Scenario", y=f"{valueName}", hue="SP", ax=axes[index], data=pdData,
                       split=True, palette="Set2", cut=0,
                       inner="quartile", width=0.6, linewidth=1)

    for artist in axes[index].lines:
        artist.set_zorder(10)
    for artist in axes[index].findobj(PathCollection):
        artist.set_zorder(11)

    h = sns.stripplot(x="Scenario", y=f"{valueName}", hue="SP", data=pdData,
                      dodge=True, jitter=True, ax=axes[index], palette="Set2", linewidth=0.6, size=6)
    axes[index].set_ylim([-2, 22])

    if index == 6:
        axes[index].legend(handles=axes[index].legend_.legendHandles, labels=['SPO', 'DPO', 'SPO', 'DPO'],
                           ncol=2, fancybox=True, loc='lower center', bbox_to_anchor=(0.5, -0.5))
    else:
        axes[index].legend([], [], frameon=False)


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
