import csv

import pandas as pd
import numpy as np
import os


# sorts Beiwe csv info into a dictionary
def clean_file(filename):
    dat = pd.read_csv(filename)
    dat = dat.filter(["hashed MAC", " RSSI"])
    dat = dat.sort_values("hashed MAC")
    dat = np.asarray(dat)
    data_dict = {}
    for i in range(dat.shape[0]):
        data_dict[dat[i, 0]] = dat[i, 1]
    return data_dict


# fill missing mac addresses with -120
def fill_missing_macs(raw_data, macs):
    rssi = []
    for i in range(macs.shape[0]):
        if raw_data.get(macs[i]) is not None:
            rssi.append(raw_data[macs[i]])
        else:
            rssi.append(-120)
    filled_data = np.asarray(rssi)
    return filled_data


# preprocess data from Beiwe
def preprocess(known_file, folder):
    mac = np.asarray(pd.read_csv(known_file).columns[1:])
    for k in range(mac.shape[0]):
        m = mac[k]
        if m[0] == ' ':
            mac[k] = m[1:]
    files = os.listdir(folder)
    rssi_series = []
    num_files = len(files)
    percent = int(num_files / 100)
    n = 0
    stamps = []
    for f in files:
        raw = clean_file(folder + "/" + f)
        filled = fill_missing_macs(raw, mac)
        rssi_series.append(filled)
        stamps.append(f[-17: -4])
        n = n+1
        if n % percent == 0:
            print(f'{n / percent} percent complete')
    out_file = open(folder + '_rssi.csv', 'w', newline='')
    wr = csv.writer(out_file, delimiter=',')
    mac = mac.tolist()
    mac.append('times')
    wr.writerow(mac)
    i = 0
    for s in range(len(rssi_series)):
        rssi_ser = rssi_series[s].tolist()
        rssi_ser.append(stamps[i])
        wr.writerow(rssi_ser)
        i = i+1


# preprocess data from the data collection app
def preprocess_app(known_file, data_file):
    mac = np.asarray(pd.read_csv(known_file).columns[1:])
    for k in range(mac.shape[0]):
        m = mac[k]
        if m[0] == ' ':
            mac[k] = m[1:]
    times = []
    rssi_series = []
    with open(data_file) as csv_file:
        d_reader = csv.reader(csv_file, delimiter=',')
        for row in d_reader:
            times.append(row[0])
            pairs = {}
            for k in range(1, len(row)):
                if len(row[k]) > 1:
                    vals = row[k].split('=')
                    pairs[vals[0]] = int(vals[1])
            filled_data = fill_missing_macs(pairs, mac)
            rssi_series.append(filled_data)
    out_file = open(data_file[:-4] + '_rssi.csv', 'w', newline='')
    wr = csv.writer(out_file, delimiter=',')
    mac = mac.tolist()
    mac.append('times')
    wr.writerow(mac)
    i = 0
    for s in range(len(rssi_series)):
        rssi_ser = rssi_series[s].tolist()
        rssi_ser.append(times[i])
        wr.writerow(rssi_ser)
        i = i + 1


# combines list of csv fingerprint files into one combined csv fingerprint
def combine_prints(csv_prints, file_name="combined_prints.csv"):
    master_frame = pd.read_csv(csv_prints[0])
    for csv_file in csv_prints[1:]:
        prints = pd.read_csv(csv_file)
        master_frame = master_frame.append(prints)
    master_frame.fillna(-120, inplace=True)
    master_frame.to_csv(file_name, index=False)


if __name__ == '__main__':
    fold = "3qu5abkp"
    known = "combined_prints.csv"
    # preprocess(known, fold)
    preprocess_app(known, 'validation/validate_13.csv')
    # print_files = ["fingerprint_1_64.csv", "fingerprint_72_102.csv", "fingerprint_103_245.csv",
    #                "fingerprint_245_354.csv"]
    # combine_prints(print_files)
