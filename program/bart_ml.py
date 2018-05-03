#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Libao Jin'
__date__ = '05/02/2018'
__email__ = 'ljin1@uwyo.edu'


import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydot


class BartClassifier(object):

    def __init__(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def load_data(self, filenames):
        '''Load datasets in batch'''
        datasets = []
        for filename in filenames:
            data = pd.read_csv(filename)
            datasets.append(data)
        print('Data loaded.')
        return datasets

    def data_preparation(self, bart, stat):
        loc = stat['Location'].str.split(',', expand=True)
        loc = [pd.to_numeric(loc[i]) for i in loc.columns]
        stat['Longitude'], stat['Latitude'] = loc[0], loc[1]
        stat_tmp = stat.copy()
        stat_tmp.index.names = ['Stat_ID']
        stat_tmp.reset_index(level=0, inplace=True)
        stat_id = stat_tmp.set_index('Abbreviation')['Stat_ID'].dropna()
        stat_lon = stat_tmp.set_index('Abbreviation')['Longitude'].dropna()
        stat_lat = stat_tmp.set_index('Abbreviation')['Latitude'].dropna()
        bart = bart.drop(bart[(bart['Origin'] == 'WSPR') | (bart['Destination'] == 'WSPR')].index)
        bart['DateTime'] = pd.to_datetime(bart['DateTime'])
        bart['DayOfWeek'] = bart['DateTime'].dt.weekday
        bart['Month'] = bart['DateTime'].dt.month
        bart['Day'] = bart['DateTime'].dt.day
        bart['Hour'] = bart['DateTime'].dt.hour
        print('First session done.')
        bart['OriginID'] = bart['Origin'].replace(stat_id)
        bart['OriginLongitude'] = bart['Origin'].replace(stat_lon)
        bart['OriginLatitude'] = bart['Origin'].replace(stat_lat)
        print('Second session done.')
        bart['DestinationID'] = bart['Destination'].replace(stat_id)
        bart['DestinationLongitude'] = bart['Destination'].replace(stat_lon)
        bart['DestinationLatitude'] = bart['Destination'].replace(stat_lat)
        bart['ThroughputLevel'] = bart['Throughput'].apply(self.throughput_level)
        print('Data prepared.')
        return (stat, bart)

    def throughput_level(self, throughput):
        if throughput <= 5:
            level = 0
        elif throughput <= 15:
            level = 1
        elif throughput <= 30:
            level = 2
        elif throughput <= 80:
            level = 3
        else:
            level = 4
        return level

    def generate_train_test(self, data, feature_keys, target_keys):
        train_features, test_features, train_labels, test_labels = train_test_split(
            data[feature_keys],
            data[target_keys],
            test_size=0.99998
        )
        return (train_features, test_features, train_labels, test_labels)

    def decision_tree(self, data, feature_keys, target_keys, filename='./output/bart_dt.pdf'):
        train_features, test_features, train_labels, test_labels = self.generate_train_test(
            data, feature_keys, target_keys)
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_features, train_labels)
        test_labels_predict = clf.predict(test_features)
        hit_rate = sum([1 for i in range(len(test_labels)) if test_labels_predict[i] ==
                        test_labels.values[i]]) / len(test_labels) * 100
        print('Decision Tree: Prediction hit/accuracy rate: {:.2f}%'.format(hit_rate))
        self.visualize(clf, feature_keys, target_keys, filename)
        train_data, test_data = pd.DataFrame(), pd.DataFrame()
        train_data[feature_keys], train_data[target_keys], test_data[feature_keys], test_data[target_keys] = \
            train_features, train_labels, test_features, test_labels
        test_data = test_data.reset_index(drop=True)
        test_labels_predict = pd.DataFrame(data={'predict': test_labels_predict})
        test_data['predict'] = test_labels_predict
        train_data.to_csv('../data/training_data.csv', encoding='utf-8', index=False)
        test_data.to_csv('../data/test_data.csv', encoding='utf-8', index=False)
        return (test_features, test_labels, test_labels_predict, hit_rate)

    def visualize(self, clf, feature_keys, target_keys, filename):
        feature_names = feature_keys
        target_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        dot_data = StringIO()
        tree.export_graphviz(
            clf,
            out_file=dot_data,
            feature_names=feature_names,
            class_names=target_names,
            filled=True,
            rounded=True,
            impurity=False
        )
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf(filename)

    def linear_regression(self, data, feature_keys, target_keys, filename='./output/bart_lr.pdf'):
        lm = linear_model.LinearRegression()
        X = data[feature_keys]
        y = data[target_keys]
        lm.fit(X, y)
        predictions = lm.predict(X)
        print('Linear Regression: Coefficients: {}\nIntercept: {}\nScore: {}'.format(lm.coef_,
                                                                                     lm.intercept_,
                                                                                     lm.score(X, y)))

        with PdfPages(filename) as pdf:
            fig, ax = plt.subplots()
            ax.scatter(y, predictions, edgecolors=(0, 0, 0))
            ax.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], 'b--', lw=2)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            plt.show(block=False)
            pdf.savefig(fig)

    def run(self):
        # filenames = [
        #     '../data/bart-ridership/date-hour-soo-dest-2016.csv',
        #     '../data/bart-ridership/date-hour-soo-dest-2017.csv',
        #     '../data/bart-ridership/station_info.csv'
        # ]

        # datasets = self.load_data(filenames)
        # bart = pd.concat(datasets[0:2], ignore_index=True)
        # stat = datasets[2]

        filenames = [
            # '../data/bart-ridership/date-hour-soo-dest-2016.csv',
            '../data/bart-ridership/date-hour-soo-dest-2017.csv',
            '../data/bart-ridership/station_info.csv'
        ]
        datasets = self.load_data(filenames)
        bart = datasets[0]
        stat = datasets[1]

        stat, bart = self.data_preparation(bart, stat)
        print(stat.head())
        print(bart.head())

        # feature_keys = [
        #     'OriginID', 'OriginLongitude', 'OriginLatitude',
        #     'DestinationID', 'DestinationLongitude', 'DestinationLatitude',
        #     'Month', 'Day', 'Hour', 'DayOfWeek'
        # ]

        feature_keys = [
            'OriginID',
            'DestinationID',
            'Hour', 'DayOfWeek'
        ]

        target_keys = ['ThroughputLevel']

        self.decision_tree(bart, feature_keys, target_keys)
        self.linear_regression(bart, feature_keys, target_keys)


if __name__ == '__main__':
    bc = BartClassifier()
    bc.run()
