import csv

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocess import preprocess, preprocess_app


def validate(folds=5, file='known_positions.csv'):
    known_positions = pd.read_csv(file)
    splits = []
    for n in range(folds):
        splits.append(known_positions.iloc[n::5])
    err = 0
    for n in range(folds):
        test_data = splits[n]
        train_data = []
        for k in range(folds):
            if k != n:
                train_data.append(splits[k])
        train_data = pd.concat(train_data)
        train_y = train_data["position"]
        train_x = train_data.drop(columns=["position"])
        test_y = np.asarray(test_data["position"])
        test_x = test_data.drop(columns=["position"])

        knn = KNeighborsClassifier(n_neighbors=folds).fit(train_x, train_y)
        y_out = knn.predict(test_x)
        num_correct = 0
        for k in range(len(test_y)):
            if y_out[k, ] == test_y[k, ]:
                num_correct = num_correct + 1
        percentage = num_correct / len(test_y)
        err = err + percentage
    err = err / folds
    return err


class KNN:

    def __init__(self, file='known_positions.csv', neighbors=10):
        # load in known positions
        known_positions = pd.read_csv(file)

        # extract labels from dataset
        positions = known_positions["position"]

        # collect features in single array
        networks = known_positions.drop(columns=["position"]).apply(lambda x: 10 ** ((-60 - x)/20))

        # PCA (doesn't help)
        # self.pca = PCA(n_components=50)
        # self.pca.fit(networks)
        # networks = self.pca.transform(networks)

        # fit knn
        self.trained_knn = KNeighborsClassifier(n_neighbors=neighbors).fit(networks, positions)

    def test_locations(self, locations):
        # test new input
        locs = pd.DataFrame(locations).apply(lambda x: 10 ** ((-60 - x)/20))
        # locs = self.pca.transform(locs)
        pos = self.trained_knn.predict(locs)
        return pos


class OtherModel:

    def __init__(self, file='known_positions.csv'):
        # load in known positions
        known_positions = pd.read_csv(file)

        # extract labels from dataset
        positions = known_positions["position"]

        # collect features in single array
        networks = known_positions.drop(columns=["position"]).apply(lambda x: 10 ** ((-60 - x)/20))

        # fit knn
        self.trained_model = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(networks, positions)

    def test_locations(self, locations):
        # test new input
        locs = pd.DataFrame(locations).apply(lambda x: 10 ** ((-60 - x)/20))
        pos = self.trained_model.predict(locs)
        return pos


def process(loc_file, fol):
    # load known positions and train model
    net = KNN(file=f, neighbors=19)
    # net = OtherModel(file=f)
    # print(validate(folds=10, file=f))
    # load test positions
    df = pd.read_csv(loc_file)
    # remove useless data
    stamps = df['times']
    df = df.drop(columns=['times'])
    df = df[(df.T != -120).any()]
    indexes = df.index.values.tolist()
    location_data = np.asarray(df)
    # convert rssi into locations
    out = net.test_locations(location_data)
    # split data into sequences
    sequences = []
    time_sequences = []
    current_sequence = [out[0]]
    current_times = [stamps[indexes[0]]]
    for i in range(len(indexes) - 1):
        if indexes[i] + 1 == indexes[i + 1]:
            current_sequence.append(out[i + 1])
            current_times.append(stamps[indexes[i + 1]])
        else:
            sequences.append(current_sequence)
            current_sequence = [out[i + 1]]
            time_sequences.append(current_times)
            current_times = [stamps[indexes[i + 1]]]
    sequences.append(current_sequence)
    time_sequences.append(current_times)
    # print(sequences)
    # print(time_sequences)
    # save sequences to a file
    out_file = open('location_sequences_' + fol + '.csv', 'w', newline='')
    wr = csv.writer(out_file, delimiter=',')
    for s in range(len(sequences)):
        wr.writerow(sequences[s])
        wr.writerow(time_sequences[s])
    out_file.close()
    np.savetxt('output.csv', np.int_(out), fmt='%2d', delimiter=",")


if __name__ == '__main__':
    f = "combined_prints.csv"  # csv fingerprint file
    # print(validate(file=f))
    folder = "qpagaxi7_1"  # name of the folder/participant from Beiwe
    person = "validate_14"  # person is the file name from the data collection app (without .csv)

    # use these two lines for data from our app
    preprocess_app(f, person + ".csv")
    process(person + '_rssi.csv', person)

    # use these two lines for data from Beiwe
    # preprocess(f, folder)
    # process(folder + '_rssi.csv', folder)
