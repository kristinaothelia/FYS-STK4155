"""
FYS-STK4155 - Project 2
Various plot functions
"""
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
import scikitplot        as skplt
import functions         as func


def map():
    sns.set()

    arr     = np.load('acc_score.npy', allow_pickle=True)
    etas    = np.load('eta_values.npy', allow_pickle=True)
    lambdas = np.load('lambda_values.npy', allow_pickle=True)

    etas = ["{:0.2e}".format(i) for i in etas]

    new_arr = np.zeros(shape=arr.shape)

    Ni = len(arr[:,0])
    Nj = len(arr[0,:])

    for i in range(Ni):
        for j in range(Nj):
            new_arr[i][j] = arr[i][j]

    ax = sns.heatmap(new_arr, xticklabels=lambdas, yticklabels=etas, annot=True, linewidths=.3, linecolor="black")
    plt.title('Accuracy')
    plt.ylabel('$\\eta$')
    plt.xlabel('$\\lambda$')
    plt.tight_layout()
    plt.show()


def Cumulative_gain_plot(y_test, model):

	p = func.probabilities(model)
	notP = 1 - np.ravel(p)
	y_p = np.zeros((len(notP), 2))
	y_p[:,0] = np.ravel(p)
	y_p[:,1] = np.ravel(notP)

	x_plot, y_plot = func.bestCurve(y_test)

	skplt.metrics.plot_cumulative_gain(y_test, y_p, text_fontsize='medium')
	plt.plot(x_plot, y_plot, linewidth=4)
	plt.legend(["Pay", "Default", "Baseline", "Best curve"])
	plt.ylim(0, 1.05)
	plt.show()

def ROC_plot(y_test, predict_proba_scikit):
	skplt.metrics.plot_roc(y_test, predict_proba_scikit)
	plt.show()

def Confusion_matrix(y_test, model):

	CM 			 = func.Create_ConfusionMatrix(model, y_test, plot=True)
	CM_DataFrame = func.ConfusionMatrix_DataFrame(CM, labels=['pay', 'default'])

	print('-------------------------------------------')
	print('The Confusion Matrix')
	print(CM_DataFrame)
	print('-------------------------------------------')


def Histogram(category, name, x_label):

    plt.hist(category, bins=900, color="purple")

    plt.title("Histogram for " + name)
    plt.xlabel(x_label)
    plt.ylabel("Observations count")
    plt.xlim(0, 5000)
    #plt.savefig("Hist/"+name+".png")
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
    plt.show()

def Multi_hist(list_, name, label, title_):

    ncols = 3; nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    counter = 0

    for i in range(nrows):
        for j in range(ncols):

            ax = axes[i][j]
            if counter < 7:

                #ax.hist(list_[counter], bins='auto', color='purple')
                #ax.set_xlabel("Repayment status %s" %name[counter], fontsize=15)
                #ax.set_ylabel("Observations count", fontsize=15)
                #ax.set_title(name[counter], fontsize=15)

                if list_ == "list_PAY":
                    ax.hist(list_[counter], bins='auto', color='purple')
                    ax.set_xlabel("Repayment status %s" %name[counter], fontsize=15)
                    ax.set_ylabel("Observations count", fontsize=15)
                    ax.set_title(name[counter], fontsize=15)
                else:
                    ax.hist(list_[counter], bins='auto', color='purple')
                    ax.set_xlabel(label+"%s" %(counter), fontsize=15)
                    ax.set_ylabel("Observations count", fontsize=15)
                    ax.set_title(title_+"%s" %(counter), fontsize=15)

            # Remove axis when we no longer have data
            else:
                ax.set_axis_off()

            counter += 1
    plt.tight_layout(w_pad=-5, h_pad=-1)
    plt.show()
