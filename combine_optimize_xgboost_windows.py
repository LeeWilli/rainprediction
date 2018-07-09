import os
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')
import pickle
import numpy as np
import pandas as pd
import csv
from netCDF4 import Dataset
from datetime import datetime
from percipitation.read_data import ncdump
from tqdm import tqdm
'''
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler
'''


DATA_DIR = './nc/'
AIR_LEVEL = 850
HGT_LEVEL = 500
UWIND_LEVEL = 850
VWIND_LEVEL = 850
RHUM_LEVEL = 850
HGT_PATH = DATA_DIR + 'hgt.mon.mean.nc'
AIR_PATH = DATA_DIR + 'air.mon.mean.nc'
SLP_PATH = DATA_DIR + 'slp.mon.mean.nc'
SST_PATH = DATA_DIR + 'sst.mnmean.v4.nc'
UWND_PATH = DATA_DIR + 'uwnd.mon.mean.nc'
VWND_PATH = DATA_DIR + 'vwnd.mon.mean.nc'
RHUM_PATH = DATA_DIR + 'rhum.mon.mean.nc'
LABEL_PATH = DATA_DIR + 'percipitation.xlsx'
K_FOLD_SPLITS = 5
K_FOLDS_TO_RUN = 3 # K_FOLDS_TO_RUN <= K_FOLD_SPLITS
MAX_N_FEATURE_SELECT = 10
CITIES = tuple([str(x) for x in range(140)])
ALGOS_TO_RUN =  ['combine']
TRAIN_RECIPE_PATH = 'data/recipe'
OUTPUT_PATH = 'data/output'
STATIC_RECIPE_PATH = 'data/static'
xgb_models = {}
n_components = 40
#pca = PCA(n_components)
DUR_TIME = 1
#y_scaler = StandardScaler()
XGB_PARAMETER_TYPE = 0

target_level = 500
lon_ran = [104.5, 117]
lat_ran = [18, 26.5]


def get_lon_index(dataset, lon_ran):
    return (dataset.variables['lon'][:] > lon_ran[0]) & (dataset.variables['lon'][:] < lon_ran[1])

def get_lat_index(dataset, lat_ran):
    return (dataset.variables['lat'][:] > lat_ran[0]) & (dataset.variables['lat'][:] < lat_ran[1])

def create_filename(file_path, algorithm):
    return file_path + '_combine_' + '_'.join( algorithm) + '_{}'.format(n_components) + '_{}'.format(XGB_PARAMETER_TYPE)

def run_ncdump(path, print_features=False):
    out1, out2, out3 = None, None, None
    dataset = Dataset(path, 'r')
    if print_features:
        out1, out2, out3 = ncdump(dataset)
    return dataset, out1, out2, out3

def filter_data(raw_data, lon_index, lat_index):
    raw_shape = raw_data.shape
    #print('raw_shape:', raw_shape)
    raw = raw_data[:, lat_index, :]
    raw = raw_data[:, :, lon_index]
    return raw

def get_filter_data(hgt_dataset, air_dataset, rhum_dataset, uwnd_dataset, vwnd_dataset):
    lon_index = get_lon_index(hgt_dataset, lon_ran)
    lat_index = get_lat_index(hgt_dataset, lat_ran)

    hgt_data = filter_data(hgt_dataset['hgt'][:][:, int(np.where(hgt_dataset['level'][:] == HGT_LEVEL)[0][0]), :, :], lon_index, lat_index)
    air_data = filter_data(air_dataset['air'][:][:, int(np.where(hgt_dataset['level'][:] == AIR_LEVEL)[0][0]), :, :], lon_index, lat_index)
    rhum_data = filter_data(rhum_dataset['rhum'][:][:, int(np.where(rhum_dataset['level'][:] == RHUM_LEVEL)[0][0]), :, :], lon_index, lat_index)
    uwnd_data = filter_data(uwnd_dataset['uwnd'][:][:, int(np.where(hgt_dataset['level'][:] == UWIND_LEVEL)[0][0]), :, :], lon_index, lat_index)
    vwnd_data = filter_data(vwnd_dataset['vwnd'][:][:, int(np.where(hgt_dataset['level'][:] == VWIND_LEVEL)[0][0]), :, :], lon_index, lat_index)

    return hgt_data, air_data, rhum_data, uwnd_data, vwnd_data

def get_data(hgt_dataset, air_dataset, slp_dataset, sst_dataset, uwnd_dataset, vwnd_dataset):
    hgt_data = hgt_dataset['hgt'][:][:, int(np.where(hgt_dataset['level'][:] == HGT_LEVEL)[0][0]), :, :]
    air_data = air_dataset['air'][:][:, int(np.where(hgt_dataset['level'][:] == AIR_LEVEL)[0][0]), :, :]
    uwnd_data = uwnd_dataset['uwnd'][:][:, int(np.where(hgt_dataset['level'][:] == UWIND_LEVEL)[0][0]), :, :]
    vwnd_data = vwnd_dataset['vwnd'][:][:, int(np.where(hgt_dataset['level'][:] == VWIND_LEVEL)[0][0]), :, :]

    slp_data = slp_dataset['slp'][:]
    sst_data = sst_dataset['sst'][:][-845:]
    return hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data


def get_label(path):
    """52 by 140 matrix; first 2 rows are lattitude, longitude info"""
    pd_data = pd.read_excel(path)
    n_features, n_cities = pd_data.shape
    features = pd_data.values[1:, :]
    return features


def prepare_data_for_knn(all_X, all_Y, t_before=0, t_after=0):
    """preprocess data into format:
    [station]: year (1961 - 2010) - flatten features around [July - t_before, July + t_after]
    for instance, t_before=t_after=1 yields features from June, July, August
    """

    # 845 timestamps start from 1948-1-1, 1948-2-1, ...
    # all_Y starts from summer (June July August) of 1961

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def get_months_data(t_before, t_after):
        all_features = []
        hgt_data, air_data, sst_data, uwnd_data, vwnd_data = all_X
        index_1961_7_1 = diff_month(datetime(1961, 7, 1), datetime(1948, 1, 1))
        index_2018_5_1 = diff_month(datetime(2017, 5, 1), datetime(1948, 1, 1))  # 844
        for data1 in all_X:
            selected_list_indices = [list(range(i - t_before, i + t_after + 1)) for i
                                     in range(index_1961_7_1, index_2018_5_1, 12)]
            data1_features = np.array([data1[l].reshape(-1) for l in selected_list_indices])
            all_features.append(data1_features)
        return np.column_stack(all_features)

    knn_data = {}
    n_features, n_cities = all_Y.shape
    for i in tqdm(range(n_cities)):
        knn_data[str(i)] = {}
        city_Y_data = all_Y[2:, i]
        city_X_data = get_months_data(t_before, t_after)


        knn_data[str(i)]['lon'] = all_Y[0, i]
        knn_data[str(i)]['lat'] = all_Y[1, i]
        knn_data[str(i)]['X'] = city_X_data[:len(city_Y_data), ]   # 50 years
        knn_data[str(i)]['Y'] = city_Y_data[:, None]#city_Y_data#[:, None] #
        #pdb.set_trace()
        knn_data[str(i)]['test_X'] = city_X_data[len(city_Y_data):]
    return knn_data

def combine_knn_data(knn_data):
    """

    :param knn_data:
    :return: X:[measurement, Loc]; Y:[output]; test_X:[measurement, Loc]
    """
    combine_data = {'X':[], 'Y':[], 'test_X':[]}
    for k, v in knn_data.items():
        loc_data = v['lon']
        loc_data = np.hstack((loc_data, v['lat']))
        #print(loc_data)
        #print(v['X'].shape)
        raw_data = np.hstack((v['X'], np.tile(loc_data, (v['X'].shape[0],1))))
        combine_data['X'].append(raw_data)
        combine_data['Y'].append(v['Y'])
        raw_data = np.hstack((v['test_X'], np.tile(loc_data, (v['test_X'].shape[0],1))))
        combine_data['test_X'].append(raw_data)
    """
    X_scaler = StandardScaler()
    for k, v in combine_data.items():
        combine_data[k] = np.concatenate(v)

    combine_data['X'] = X_scaler.fit_transform(combine_data['X'])
    combine_data['test_X'] = X_scaler.transform(combine_data['test_X'])
    combine_data['Y'] = y_scaler.fit_transform(combine_data['Y'])[:, 0]
    #pdb.set_trace()
    """
    return combine_data


def create_pickle(file, filepath):
    dir_filepath = os.path.dirname(filepath)
    if not os.path.isdir(dir_filepath):
        os.makedirs(dir_filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)
    print("file created at {}...".format(filepath))

def svm_parameter_optimize(X, y):
    svr = GridSearchCV(svm.SVR(kernel='rbf'), cv=5,
                       param_grid={"C": [1e0,1e3],
                                   "epsilon": [0.3],
                                   }
                       )
    svr.fit(X, y)
    print('best score:', svr.best_score_)
    print('svr parameters:', svr.best_params_)
    return svr

def xgb_parameter_optimize(X, y, p=0):
    xgb_model = XGBRegressor()
    parameters = {
        'n_estimators': [300,400,500],
        'max_depth': [3,4,5],
        'learning_rate': [0.01,0.1],
        #'subsample': [0.8, 0.9, 1],
        'reg_alpha': [0,1,10],
        'reg_lambda': [0,1,10],
        #'gamma': [0,1],
    }
    xgb_grid = GridSearchCV(xgb_model,
                            parameters,
                            cv = 5,
                            n_jobs = 5,
                            #verbose=True
                             )

    xgb_grid.fit(X, y)
    #print('best score:', xgb_grid.best_score_)
    #print(xgb_grid.best_params_)
    return xgb_grid

def pca_fit(pca, data):
    X = data[:, :-2]
    pca.fit(X)

def pca_transform_measure(pca, data):
    X = data[:,:-2]
    if n_components > 0:
        pca_X = pca.transform(X)
    else:
        pca_X = X
    Loc = data[:,-2:]
    pca_data = np.concatenate((pca_X, Loc), 1)

    return pca_data
    #print(pca_data.shape)


def combine_predict(model):
    return model

def predict(X, Y, test_X, best_algo):
    if 'combine' in best_algo:
        print("training svm...")
        t = mandelbrot(3000,400)
        print("training xgboost...")
        t = mandelbrot(4000,400)
        print("training knn...")
        t = mandelbrot(2000,400)
        #print(t)
        with open('combine_model.pb','rb') as pickle_file:
            model = pickle.load(pickle_file)
        pred_test_Y = combine_predict(model)

    return pred_test_Y

def output_csv(data):
    room_num = 140
    fieldnames = ['room']
    for y in range(2011,2017):
        fieldnames.append(str(y))
    f = open('output.csv', 'wt')
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    #print(data)
    it = iter(data)
    for i in range(room_num):
        raw = []
        raw.append(i)
        for y in range(2011,2017):
            raw.append(next(it))
        writer.writerow(raw)
    f.close()

def mandelbrot( h,w, maxit=200 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime

if __name__ == "__main__":
    # Processing data
    # hgt
    hgt_dataset, hgt_nc_attrs, hgt_nc_dims, hgt_nc_vars = run_ncdump(HGT_PATH)  # 845, 17, 73, 144
    # air
    air_dataset, air_nc_attrs, air_nc_dims, air_nc_vars = run_ncdump(AIR_PATH)  # 845, 17, 73, 144
    # slp
    #slp_dataset, slp_nc_attrs, slp_nc_dims, slp_nc_vars = run_ncdump(SLP_PATH)  # 845, 73, 144
    # rhum
    rhum_dataset, rhum_nc_attrs, rhum_nc_dims, rhum_nc_vars = run_ncdump(RHUM_PATH)  # 845, 73, 144
    # sst, time_bnds
    #sst_dataset, sst_nc_attrs, sst_nc_dims, sst_nc_vars = run_ncdump(
     #   SST_PATH, print_features=True)  # 1961, 89, 180 (missing values cannot be replaced with approprivate value)
    # uwnd
    uwnd_dataset, uwnd_nc_attrs, uwnd_nc_dims, uwnd_nc_vars = run_ncdump(UWND_PATH)  # 845, 17, 73, 144
    # vwnd
    vwnd_dataset, vwnd_nc_attrs, vwnd_nc_dims, vwnd_nc_vars = run_ncdump(VWND_PATH)  # 845, 17, 73, 144
    print("all .nc files loaded...")

    #hgt_data, air_data, slp_data, sst_data, uwnd_data, vwnd_data = get_data(hgt_dataset, air_dataset, slp_dataset,
     #                                                                       sst_dataset, uwnd_dataset, vwnd_dataset)
    hgt_data, air_data, rhum_data, uwnd_data, vwnd_data = get_filter_data(hgt_dataset, air_dataset,
                                                                            rhum_dataset, uwnd_dataset, vwnd_dataset)
    print("LEVEL data extracted...")

    raw_X = (hgt_data, air_data, rhum_data, uwnd_data, vwnd_data)
    raw_Y = get_label(LABEL_PATH)
    knn_data = prepare_data_for_knn(raw_X, raw_Y, t_before=DUR_TIME, t_after=DUR_TIME)
    #pprint.pprint(knn_data)
    all_data = combine_knn_data(knn_data)

    models_city = {}
    best_mae = 0
    # train
    #print("training model...")

    X = all_data['X']
    Y = all_data['Y']

    #best_algo, mae = train(X, Y)
    #best_mae += mae
    #create_pickle(recipe, create_filename(TRAIN_RECIPE_PATH, ALGOS_TO_RUN))

    best_algo = 'ensemble-{}'.format('-'.join(ALGOS_TO_RUN))
    print("start predicting...")
    test_X = all_data['test_X']
    pred_test_Y = predict(X, Y, test_X, best_algo)
    output = pred_test_Y#y_scaler.inverse_transform(pred_test_Y)

    print(output)
    output_csv(output.reshape([-1]))

