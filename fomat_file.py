import pickle
import pprint
import csv
import os
import numpy as np

#from harry_main import OUTPUT_PATH
#from optimize_xgboost import OUTPUT_PATH, ALGOS_TO_RUN, create_filename
#from combine_optimize_xgboost import OUTPUT_PATH, ALGOS_TO_RUN, create_filename

file_name = 'data/output_combine_knn_xgboost_svm_best_1.csv'
def pickle2csv0(file_name):
    room_num = 140
    #print(create_filename(file_name, algos))
    pkl_file = open(file_name, 'rb')
    result = pickle.load(pkl_file)
    for i in range(room_num):
        pprint.pprint(result[str(i)])

    fieldnames = ['room']
    for y in range(2011,2017):
        fieldnames.append(str(y))
    f = open(file_name+'.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for i in range(room_num):
        raw = []
        raw.append(i)
        for j in result[str(i)]:
            raw.append(j)
        writer.writerow(raw)
    f.close()

def pickle2csv(file_name, algos):
    room_num = 140
    print(create_filename(file_name, algos))
    pkl_file = open(create_filename(file_name, algos), 'rb')
    result = pickle.load(pkl_file)
    for i in range(room_num):
        pprint.pprint(result[str(i)])

    fieldnames = ['room']
    for y in range(2011,2017):
        fieldnames.append(str(y))
    f = open(create_filename(file_name, algos)+'.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for i in range(room_num):
        raw = []
        raw.append(i)
        for j in result[str(i)]:
            raw.append(j)
        writer.writerow(raw)
    f.close()

def combine_pickle2csv(file_name, algos):
    room_num = 140
    print(create_filename(file_name, algos))
    pkl_file = open(create_filename(file_name, algos), 'rb')
    result = pickle.load(pkl_file)
    pprint.pprint(result)

    fieldnames = ['room']
    for y in range(2011,2017):
        fieldnames.append(str(y))
    f = open(create_filename(file_name, algos)+'.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    it = iter(result)
    for i in range(room_num):
        raw = []
        raw.append(i)
        for y in range(2011,2017):
            raw.append(next(it))
        writer.writerow(raw)
    f.close()

def create_pickle(file, filepath):
    dir_filepath = os.path.dirname(filepath)
    if not os.path.isdir(dir_filepath):
        os.makedirs(dir_filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)

def combine_csv2pickle(file_name):
    room_num = 140
    outputs=[]
    f = open(file_name, 'rt')
    reader = csv.reader(f)
    first = True
    for row in reader:
        if first == True:
            first = False
        else:
            outputs.append(row[1:])
    print(outputs)
    a = np.array(outputs, dtype=np.float32)
    #print(a)
    #print(a.shape)
    create_pickle(a, './combine_model.pb')

def load_pickle(file_name):
    room_num = 140
    pkl_file = open(file_name+'.pb', 'rb')
    result = pickle.load(pkl_file)
    pprint.pprint(result)


if __name__ == "__main__":
    #pickle2csv0(OUTPUT_PATH)
    #pickle2csv(OUTPUT_PATH, ALGOS_TO_RUN)
    #combine_pickle2csv(OUTPUT_PATH, ALGOS_TO_RUN)
    combine_csv2pickle(file_name)
    #load_pickle(file_name)

    with open('combine_model.pb', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
        print(model)

