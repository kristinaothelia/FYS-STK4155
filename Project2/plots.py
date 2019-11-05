"""
FYS-STK4155 - Project 2
Various plot functions
"""

import numpy             as np
import matplotlib.pyplot as plt

def Histogram(category, name, x_label):

    plt.hist(category, bins=900, color="purple")

    plt.title("Histogram for " + name)
    plt.xlabel(x_label)
    plt.ylabel("Observations count")
    plt.xlim(0, 5000)
    #plt.savefig("Hist/"+name+".png")
    plt.show()

def Multi_hist(list_, name, label, title_, diff=False):

    ncols = 3; nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    counter   = 0

    for i in range(nrows):
        for j in range(ncols):

            ax = axes[i][j]
            if counter < 7:

                if diff:
                    ax.hist(list_[counter], bins='auto', color='purple')
                    ax.set_xlabel("Repayment status %s" %name[counter], fontsize=15)
                    ax.set_ylabel("Observations count", fontsize=15)
                    ax.set_title(name[counter], fontsize=15)
                else:
                    ax.hist(list_[counter], bins='auto', color='purple')
                    ax.set_xlabel(label, fontsize=15)
                    ax.set_ylabel("Observations count", fontsize=15)
                    ax.set_title(title_+"%g" %(counter+1), fontsize=15)

                    for ax in fig.axes:
                        plt.sca(ax)
                        plt.xticks(rotation=20, ha='center')

            # Remove axis when we no longer have data
            else:
                ax.set_axis_off()

            counter += 1

    plt.tight_layout(w_pad=-5, h_pad=-1)
    plt.show()

def Hist_Sex_Marriage_Education(category, name):
    """
    Histograms made for the classes SEX, MARRIAGE and EDUCATION.
    Done by using plt.bar, insted of plt.hist as in the Histogram() function
    """
    if name == "SEX":
        bar_tick_label = ["Male", "Female"]
        bar_label      = ["Male", "Female"]
    elif name == "Marriage":
        bar_tick_label = ["Married", "Singel", "Other"]
        bar_label      = ["Married", "Singel", "Other"]
    elif name == "Education":
        bar_tick_label = ["Graduate school", "University", "High school", "Others"]
        bar_label      = ["Graduate school", "University", "High school", "Others"]


    fig, ax = plt.subplots()
    labels, counts = np.unique(category, return_counts=True)
    bar_plot = plt.bar(labels,counts,tick_label=bar_tick_label, width=0.4, color="purple")

    def autolabel(rects):
        for idx,rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                    bar_label[idx], ha='center', va='bottom')

    #autolabel(bar_plot)
    plt.title("Histogram for " + name)
    plt.xlabel(name)
    plt.ylabel("Observations count")
    #plt.savefig("Hist/"+name+".png")

###
