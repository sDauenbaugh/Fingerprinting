import csv
import scipy

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

    def __init__(self, file, neighbors=10):
        # load in known positions
        #known_positions = pd.read_csv(file)
        known_positions = file

        # extract labels from dataset
        self.positions = known_positions["positio"]

        # collect features in single array
        self.networks = known_positions.drop(columns=["positio"]).apply(lambda x: 10 ** ((-60 - x)/20))
        #self.networks = known_positions.drop(columns=["positio"]).apply(lambda x: ((x - -92)/-31-92))


        # PCA (doesn't help)
        # self.pca = PCA(n_components=50)
        # self.pca.fit(networks)
        # networks = self.pca.transform(networks)

        # fit knn
        self.trained_knn = KNeighborsClassifier(n_neighbors=neighbors, metric = 'manhattan', algorithm='brute').fit(self.networks, self.positions)


    def test_locations(self, locations):
        # test new input
        locs = pd.DataFrame(locations).apply(lambda x: 10 ** ((-60 - x)/20))
        locs = pd.DataFrame(locations).apply(lambda x: ((x - -92)/-31-92))
        # locs = self.pca.transform(locs)
        # sparse_matrix = scipy.sparse.csr_matrix(locs)
        pos = self.trained_knn.predict(locs)
        #d, idxs = self.trained_knn.kneighbors(X = sparse_matrix)
        # print(idxs)
        # print(d)
        # print(idxs.shape)
        return pos


class OtherModel:

    def __init__(self, file='known_positions.csv'):
        # load in known positions
        known_positions = pd.read_csv(file)

        # extract labels from dataset
        positions = known_positions["position"]

        # collect features in single array
        #networks = known_positions.drop(columns=["position"]).apply(lambda x: 10 ** ((-60 - x)/20))
        #networks = known_positions.drop(columns=["position"]).apply(lambda x: 10 ** ((x - )/20))

        # fit knn
        self.trained_model = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(networks, positions)

    def test_locations(self, locations):
        # test new input
        locs = pd.DataFrame(locations).apply(lambda x: 10 ** ((-60 - x)/20))
        sparse_matrix = scipy.sparse.csr_matrix(locs)
        pos = self.trained_model.predict(sparse_matrix)
        return pos


def process(loc_file, fol):
# load known positions and train model

  df = pd.read_csv(loc_file)
  # remove useless data
  stamps = df['times']
  df = df.drop(columns=['times'])
  df = df[(df.T != -120).any()] # was -120
  
  # code to filter out non UT MAC addresses
  eer_macs = pd.read_csv("EER-AP_MAC.csv")
  eer_macs_list = eer_macs["Unnamed: 2"].tolist()
  eer_macs_list = [i[:-1].strip() for i in eer_macs_list]

  LUT = pd.read_csv(f)

  LUT.columns = [col_name[:-1].strip() for col_name in LUT.columns]
  df.columns = [col_name[:-1].strip() for col_name in df.columns]
  #display(df)

  already_deleted = []
  for column in LUT.columns:
    if column not in eer_macs_list and column not in already_deleted:
      #print(column)
      if column != "positio":
        del LUT[column]
        del df[column]
      already_deleted.append(column)

  # display(df)
  # display(LUT)

  net = KNN(LUT, neighbors=5)

  #display(net.networks)
  indexes = df.index.values.tolist()
  location_data = np.asarray(df)
  # convert rssi into locations
  out = net.test_locations(location_data)
  display(out)
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

  # save sequences to a file
  out_file = open('location_sequences_' + fol + '.csv', 'w', newline='')
  wr = csv.writer(out_file, delimiter=',')
  for s in range(len(sequences)):
      wr.writerow(sequences[s])
      wr.writerow(time_sequences[s])
  out_file.close()
  np.savetxt('output.csv', np.int_(out), fmt='%2d', delimiter=",")
  #print(LUT.loc[125].tolist())


if __name__ == '__main__':
    f = "combined_prints.csv"  # csv fingerprint file
    # print(validate(file=f))
    folder = "qpagaxi7_1"  # name of the folder/participant from Beiwe
    person = "210425-2347_testerr1"  # person is the file name from the data collection app (without .csv)

    # use these two lines for data from our app
    preprocess_app(f, person + ".txt")
    process(person + '_rssi.csv', person)

    # use these two lines for data from Beiwe
    # preprocess(f, folder)
    # process(folder + '_rssi.csv', folder)