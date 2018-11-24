from analysis.helper import *
import scipy.stats as stats
import os

def plot_prob_response():
    """
    Calculate the probability of response
    :param nwbfile:
    :return:
    """
    filenames = get_nwbfile_names("../data")
    x = [i for i in range(1, 7, 1)]
    response_percentage_old, std_old, response_percentage_new, std_new = extract_probability_response(filenames,
                                                                                                      type="old")
    plt.errorbar(x, response_percentage_old, yerr=std_old, color='blue', label='old stimuli')
    plt.errorbar(x, response_percentage_new, yerr=std_new, color='red', label='new stimuli')
    plt.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
    plt.xlabel('Confidence')
    plt.ylabel('Probability of Response')
    plt.title('n=' + str(len(filenames)) + ' sessions')
    plt.show()


def plot_cumulative_roc():
    """
    Plot function for the cumulative roc
    :return:
    """
    filenames = get_nwbfile_names("../data")
    for filename in filenames:
        nwbfile = read(filename)
        stats_all = cal_cumulative_d(nwbfile)
        x = stats_all[0:5, 4]
        y = stats_all[0:5, 3]
        plt.plot(x, y, marker='.', color='grey', alpha=0.5)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
    plt.plot([0, 1], [0, 1], color='black', alpha=0.7)
    plt.xlabel('false alarm rate')
    plt.ylabel('hit rate')
    plt.title('average roc')
    plt.show()


def plot_overall_performance():
    filenames = get_nwbfile_names("../data")
    all_performances = []
    for filenames in filenames:
        nwbfile = read(filenames)
        stats_all = cal_cumulative_d(nwbfile)
        all_performances.append([stats_all[2, 4], stats_all[2, 3]])

    avg_performance = np.average(all_performances, axis=0)
    std_performance = np.std(all_performances, axis=0)

    for performance in all_performances:
        plt.plot(performance[0], performance[1], marker='.', color='grey', alpha=0.6)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
    plt.plot([0, 1], [0, 1], color='black', alpha=0.7)
    plt.errorbar(avg_performance[0], avg_performance[1], std_performance[1], std_performance[0])
    plt.xlabel('false alarm rate')
    plt.ylabel('hit rate')
    plt.title('Overall Performance mTP=' + str(avg_performance[0]) + ' mFP=' + str(avg_performance[1]))
    plt.show()


def plot_auc():
    filenames = get_nwbfile_names("../data")
    all_auc = []
    for filenames in filenames:
        nwbfile = read(filenames)
        auc = cal_auc(nwbfile)
        all_auc.append(auc)
    m_auc = np.mean(all_auc)
    plt.hist(all_auc, 15, histtype='bar')
    plt.xlim(0, 1)
    plt.xlabel('AUC')
    plt.ylabel('nr of subjects')
    plt.title('AUC m=' + str(m_auc))
    plt.show()


def plot_confidence_accuracy():
    filenames = get_nwbfile_names("../data")

    accuracies_high = []
    accuracies_low = []
    accuracies_all = []

    for filename in filenames:
        nwbfile = read(filename)
        split_status, split_mode, ind_TP_high, ind_TP_low, ind_FP_high, ind_FP_low, ind_TN_high, \
        ind_TN_low, ind_FN_high, ind_FN_low, n_response = dynamic_split(nwbfile)
        nr_TN_high = len(ind_TN_high[0])
        nr_TP_high = len(ind_TP_high[0])
        nr_TN_all = len(ind_TN_high[0]) + len(ind_TN_low[0])
        nr_TN_low = len(ind_TP_high[0]) + len(ind_TP_low[0])
        nr_TP_low = len(ind_TP_low[0])
        nr_TN_low = len(ind_TN_low[0])

        nr_high_response = len(ind_TN_high[0]) + len(ind_TP_high[0]) + len(ind_FN_high[0]) + len(ind_FP_high[0])
        nr_low_response = len(ind_TN_low[0]) + len(ind_TP_low[0]) + len(ind_FN_low[0]) + len(ind_FP_low[0])

        per_accuracy_high = (nr_TN_high + nr_TP_high) / nr_high_response
        per_accuracy_low = (nr_TN_low + nr_TP_low) / nr_low_response

        per_accuracy_all = (nr_TN_low + nr_TP_high) / n_response

        accuracies_high.append(per_accuracy_high*100)
        accuracies_low.append(per_accuracy_low*100)
        accuracies_all.append(per_accuracy_all*100)

    p1 = stats.ttest_1samp(accuracies_high, 50)[1]
    p2 = stats.ttest_1samp(accuracies_low, 50)[1]
    x_axis_label_high = 'high p=' + str(p1)
    x_axis_label_low = 'low p=' + str(p2)
    x_axis = [x_axis_label_high, x_axis_label_low]

    for i in range(len(accuracies_high)):
        plt.plot(x_axis, [accuracies_high[i], accuracies_low[i]], marker='o')
    plt.plot(x_axis, [50, 50], color='black')
    plt.ylim([0, 100])

    tstat, p_val = stats.ttest_ind(accuracies_high, accuracies_low, equal_var=False)
    plt.title('p=' + str(p_val))
    plt.xlabel('confidence p vs. 50%')
    plt.ylabel('accuracy % correct')
    plt.show()


def plot_correct_vs_confidence():
    pass









