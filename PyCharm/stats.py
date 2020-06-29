import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import channel_norm_params, NORM_AVG_TEST, f1_score, get_weights, showWeights, df_to_dict

TOTAL_SAMPLES = 31072


def class_stats(fname='../Data/full_aug.csv'):
    df = pd.read_csv(fname)
    values = df.values
    arr_cnt = np.sum(values, axis=1)
    total = values.shape[1]
    perts = arr_cnt / total

    idxs = np.argsort(cnts)[::-1]

    for idx in idxs:
        print(idx, arr_cnt[idx], round(100*perts[idx], 2))

    return arr_cnt


def origin_stats():
    full_set = df_to_dict(pd.read_csv('../Data/full.csv'))

    namelst = []
    labels = []

    for key in full_set.keys():
        namelst.append(key)
        labels.append(full_set[key])

    namelst_grouped = []
    for i in range(28):
        temp = []
        for j, item in enumerate(namelst):
            if labels[j][i] == 1:
                temp.append(item)

        namelst_grouped.append(temp)

    print('Total samples: {}'.format(len(namelst)))

    return namelst_grouped, namelst, full_set


def main_proc():
    namelst_grouped, namelst, full_set = origin_stats()

    for i, item in enumerate(namelst_grouped):
        print('--------------------------------------------')
        print(i, len(item))
        channel_norm_params(item, '../Data/train_s')


avgs = [[20.31, 14.8, 13.75, 20.86], [22.27, 14.86, 15.51, 22.69], [21.12, 12.35, 14.87, 21.61],
        [23.17, 10.13, 16.82, 23.68], [20.88, 12.06, 14.86, 21.75], [20.58, 10.67, 15.34, 21.74],
        [24.82, 20.3, 14.95, 24.64], [20.56, 11.1, 14.29, 21.06], [31.99, 10.23, 17.31, 28.64],
        [30.57, 11.61, 16.15, 30.71], [30.74, 10.23, 16.39, 32.08], [23.27, 14.94, 14.76, 23.45],
        [26.53, 20.5, 13.26, 26.83], [24.33, 15.07, 12.87, 24.09], [25.53, 23.85, 15.06, 25.72],
        [15.66, 7.54, 9.42, 13.06], [19.88, 14.12, 13.73, 21.11], [10.42, 11.85, 9.02, 10.52],
        [19.43, 11.47, 12.95, 18.59], [21.45, 11.16, 15.23, 22.17], [22.91, 9.05, 15.01, 21.43],
        [19.64, 21.42, 12.06, 20.23], [27.52, 15.8, 19.04, 28.9], [20.53, 11.84, 14.06, 21.28],
        [24.86, 10.27, 21.99, 26.28], [20.9, 19.69, 13.58, 21.37], [17.7, 9.19, 13.47, 17.41],
        [20.8, 8.59, 19.07, 23.49]]
stds = [[35.07, 27.3, 38.9, 35.17], [36.95, 27.81, 42.4, 37.35], [35.55, 24.73, 40.63, 35.44],
        [37.07, 22.15, 42.71, 37.44], [36.13, 25.93, 40.97, 37.13], [34.48, 22.53, 40.56, 35.39],
        [37.82, 32.15, 41.22, 37.87], [34.83, 22.62, 39.09, 34.92], [43.58, 19.74, 46.08, 42.54],
        [43.18, 20.0, 44.12, 45.77], [43.24, 18.49, 44.37, 46.37], [37.6, 29.05, 40.9, 37.88],
        [39.38, 31.5, 39.65, 39.95], [37.89, 25.05, 39.11, 38.42], [38.83, 34.99, 41.98, 39.28],
        [25.74, 14.2, 27.21, 22.74], [33.4, 25.27, 38.75, 35.59], [23.62, 23.14, 28.63, 22.76],
        [34.25, 22.06, 37.43, 33.22], [35.75, 21.83, 41.02, 36.38], [35.96, 19.09, 40.88, 35.21],
        [34.14, 33.29, 36.41, 34.61], [38.44, 25.3, 45.61, 38.55], [34.45, 23.15, 38.82, 34.88],
        [37.68, 21.34, 47.69, 37.63], [35.91, 31.23, 39.33, 36.11], [31.09, 20.05, 37.59, 31.13],
        [31.95, 17.26, 42.1, 34.64]]

cnts = [12885, 1254, 3621, 1561, 1858, 2513, 1008, 2822, 53, 45, 28, 1093, 688, 537, 1066, 21, 530, 210, 902, 1482,
        172, 3777, 802, 2965, 322, 8228, 328, 11]


def plot_util():
    idxs = np.argsort(cnts)[::-1]
    print(idxs)
    arr_avgs = np.array(avgs)[idxs]
    arr_cnts = np.array(cnts)[idxs]
    total = sum(cnts)
    print('Average: ', np.average(arr_avgs, axis=0))
    temp = []
    for i in range(4):
        temp.append(np.sum(arr_avgs[:, i]*arr_cnts/total))
    print('Weighted Average: ', temp)
    plt.figure()
    plt.plot(arr_cnts*100/total, 'k-', label='count')
    plt.plot(arr_avgs[:, 0]-NORM_AVG_TEST[0], 'r-', label='red ch')
    plt.plot(arr_avgs[:, 1]-NORM_AVG_TEST[1], 'g-', label='green ch')
    plt.plot(arr_avgs[:, 2]-NORM_AVG_TEST[2], 'b-', label='blue ch')
    plt.plot(arr_avgs[:, 3]-NORM_AVG_TEST[3], 'y-', label='yellow ch')
    plt.xlim([0, 30])
    plt.ylim([-5, 30])
    plt.xticks(range(28), idxs)
    plt.legend()
    plt.grid()
    plt.show()


def group_eval(pred='./performance/result_val_sort.csv', label='../Data/val.csv'):
    pred_df = pd.read_csv(pred)
    label_df = pd.read_csv(label)

    pred_v = pred_df.values
    label_v = label_df.values

    print(pred_v.shape, label_v.shape)

    row, col = pred_v.shape

    prs = []
    rrs = []

    sum_preds = np.sum(pred_v, axis=1)
    sum_labels = np.sum(label_v, axis=1)
    for i in range(row):
        cnt = 0
        for j in range(col):
            if pred_v[i][j] and label_v[i][j]:
                cnt += 1

        pr = cnt / sum_preds[i] if sum_preds[i] > 0 else 0
        rr = cnt / sum_labels[i] if sum_labels[i] > 0 else 0
        prs.append(pr)
        rrs.append(rr)

    idxs = np.argsort(cnts)[::-1]
    f1s = []
    for i, pr, rr in zip(idxs, np.array(prs)[idxs], np.array(rrs)[idxs]):
        f1 = f1_score(pr, rr)
        f1s.append(f1)
        print(i, round(100*pr, 2), round(100*rr, 2), round(100*f1, 2))

    print('Macro rr: {}, pr: {}, f1: {}.'.format(np.round(100*np.average(rrs), 2),
                                                 np.round(100*np.average(prs), 2),
                                                 np.round(100*np.average(f1s), 2)))

    cnt = 0
    for i in range(row):
        for j in range(col):
            if pred_v[i][j] and label_v[i][j]:
                cnt += 1

    pr = cnt / np.sum(pred_v)
    rr = cnt / np.sum(label_v)

    print('Micro rr: {}, pr: {}, f1: {}.'.format(round(100*rr, 2), round(100*pr, 2), round(100*f1_score(rr, pr), 2)))

    arr_perts = sum_labels[idxs]/col
    # for i in range(len(idxs)):
    #     print(idxs[i], sum_labels[idxs][i], round(100*arr_perts[i], 2))
    arr_f1s = np.array(f1s)
    arr_rrs = np.array(rrs)[idxs]
    arr_prs = np.array(prs)[idxs]
    plt.figure()
    plt.title(pred.split('/')[-1])
    plt.plot(arr_prs, 'g-', label='pr')
    plt.plot(arr_rrs, 'b-', label='rr')
    plt.plot(arr_f1s, 'r-', label='f1')
    plt.plot(arr_perts, 'k.', label='percentage')
    plt.xticks(range(28), idxs)
    plt.legend()
    plt.grid()
    plt.xlim([0, 30])
    plt.ylim([0, 1])
    plt.show()


def sort_pd(fname='../Data/val.csv'):
    val_df = pd.read_csv(fname)

    ans = pd.DataFrame()
    for key in sorted(val_df.keys()):
        ans[key] = val_df[key]

    ans.to_csv('../Data/sorted.csv', index=False)


def stat_weights(fname='../Data/tra_aug.csv'):
    df = pd.read_csv(fname)
    labels = df.values.transpose()

    W_p, W_n, N_p, N_n = get_weights(labels)

    showWeights(W_p, W_n, N_p, N_n)


if __name__ == '__main__':
    # main_proc()
    # plot_util()
    group_eval(pred='./performance/result_val_528770.csv', label='../Data/val_aug_sorted.csv')
    # sort_pd(fname='../Data/val_aug.csv')
    # stat_weights(fname='../Data/tra_aug.csv')
    # class_stats(fname='../Data/val_aug_sorted.csv')


