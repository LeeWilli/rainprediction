import os
import pickle
import multiprocessing
import numpy as np
import pandas as pd
import statsmodels.api as sm
from netCDF4 import Dataset
from datetime import datetime
from percipitation.read_data import ncdump
from tqdm import tqdm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn import svm
from sklearn.preprocessing import StandardScaler, Imputer
import tflearn
import pdb
from tensorflow import reset_default_graph

DATA_DIR = '/Users/harryxu/Downloads/data/因子/' if \
    os.environ['PWD'] == '/Users/harryxu/repos/weather-percipitation' \
    else '/home/wangli/code/projects/weather/nc/'
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
CITIES = tuple([str(x) for x in range(10)])#'0' # tuple(range(140)) # tuple(range(140)) # tuple([str(x) for x in range(3)]) + ('50', '100')
ALGOS_TO_RUN =  [ 'xgboost_optimize']#['lasso', 'xgboost', 'knn'] # lasso, xgboost, knn, stepwise
TRAIN_RECIPE_PATH = 'data/recipe'
OUTPUT_PATH = 'data/output'
STATIC_RECIPE_PATH = 'data/static'
xgb_models = {}
n_components = 40
pca = PCA(n_components)
DUR_TIME = 1
y_scaler = StandardScaler()
XGB_PARAMETER_TYPE = 1

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
    X_scaler = StandardScaler()
    for k, v in combine_data.items():
        combine_data[k] = np.concatenate(v)

    combine_data['X'] = X_scaler.fit_transform(combine_data['X'])
    combine_data['test_X'] = X_scaler.transform(combine_data['test_X'])
    combine_data['Y'] = y_scaler.fit_transform(combine_data['Y'])[:, 0]
    #pdb.set_trace()
    return combine_data


def create_pickle(file, filepath):
    dir_filepath = os.path.dirname(filepath)
    if not os.path.isdir(dir_filepath):
        os.makedirs(dir_filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)
    print("file created at {}...".format(filepath))

def xgb_parameter_optimize(X, y, p=0):
    xgb_model = XGBRegressor()
    if p == 0:
        parameters = {
            'n_estimators': [100, 300],
            'max_depth': [4, 6, 8],
            #'subsample': [0.8, 0.9, 1],
            'reg_alpha': [0.1, 1, 5, 10],
            'reg_lambda': [0.1, 1, 5, 10],
            #'gamma': [0, 0.1, 0.3, 1],
        }
    else:
        parameters = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 5, 7],
            #'learning_rate': [0.01, 0.1, 0.2],
            #'subsample': [0.8, 0.9, 1],
            'reg_alpha': [0.1, 1, 5, 10],
            'reg_lambda': [0.1, 1, 5, 10],
            #'gamma': [0, 0.1, 0.3, 1],
        }
    xgb_grid = GridSearchCV(xgb_model,
                            parameters,
                            cv = 3,
                            n_jobs = 5,
                            #verbose=True
                             )

    xgb_grid.fit(X, y)
    print('best csocre:', xgb_grid.best_score_)
    print(xgb_grid.best_params_)
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

def build_fcn_model(components):
    reset_default_graph()
    r2 = tflearn.metrics.R2()
    net = tflearn.input_data(shape=[None, components+2])# 2:lat and long
    net = tflearn.fully_connected(net, 128)
    net = tflearn.fully_connected(net, 128)
    net = tflearn.fully_connected(net, 1, activation='linear')
    net = tflearn.regression(net, loss='mean_square')

    return tflearn.DNN(net)

def train(X, Y):
    """search params and try different algs in ALGOS_TO_RUN;
    save a recipe, which contains the best algo (xgboost, ensemble, lasso, etc.)
     for the city"""
    # K-fold crossvalidation
    kfold = KFold(n_splits=K_FOLD_SPLITS)
    train_Ys, valid_Ys, train_metrics, valid_metrics, train_ensemble, valid_ensemble = {}, {}, {}, {}, {}, {}
    models = {}

    for algo in ALGOS_TO_RUN:
        train_Ys[algo] = []
        valid_Ys[algo] = []
        train_metrics[algo] = []
        valid_metrics[algo] = []
        train_ensemble = []
        valid_ensemble = []

    for kf_id, (train_indices, valid_indices) in enumerate(kfold.split(X)):
        if kf_id >= K_FOLDS_TO_RUN:
            break
        train_X = X[train_indices]
        val_X = X[valid_indices]
        train_Y = Y[train_indices]
        val_Y = Y[valid_indices]
        train_ensemble1 = []
        valid_ensemble1 = []

        # PCA
        #if 'xgboost' in ALGOS_TO_RUN or 'lasso' in ALGOS_TO_RUN or 'xgboost_optimize' or 'svm' in ALGOS_TO_RUN:

        pca_fit(pca, train_X)
        pca_train_X = pca_transform_measure(pca, train_X)
        #pdb.set_trace()
        pca_val_X = pca_transform_measure(pca, val_X)
        print('pca_train_X dimension:',pca_train_X.shape)
        print('train_Y dimension:', train_Y.shape)

        if 'lasso' in ALGOS_TO_RUN:
            # lasso
            model = Lasso(alpha=1.0)
            model.fit(pca_train_X, train_Y)
            pred_train_Y = model.predict(pca_train_X)
            pred_valid_Y = model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['lasso'].append(train_Y)
            valid_Ys['lasso'].append(val_Y)
            train_metrics['lasso'].append(fold_train_mae)
            valid_metrics['lasso'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'fcn' in ALGOS_TO_RUN:
            model = build_fcn_model(n_components)
            print(pca_train_X.shape)
            print(train_Y.reshape(-1,1).shape)
            #pdb.set_trace()
            model.fit(pca_train_X, train_Y.reshape(-1,1), n_epoch=100, batch_size=16, show_metric=True)
            pred_train_Y = model.predict(pca_train_X).ravel()
            pred_valid_Y = model.predict(pca_val_X).ravel()
            #print(pred_train_Y)
            #pdb.set_trace()

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['fcn'].append(train_Y)
            valid_Ys['fcn'].append(val_Y)
            train_metrics['fcn'].append(fold_train_mae)
            valid_metrics['fcn'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'SvmOptimize' in ALGOS_TO_RUN:
            svr = GridSearchCV(svm.SVR(kernel='rbf'), cv=3,
                               param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                           "gamma": np.logspace(-2, 2, 5)})
            svr.fit(pca_train_X, train_Y)
            pred_train_Y = svr.predict(pca_train_X)
            pred_valid_Y = svr.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['SvmOptimize'].append(train_Y)
            valid_Ys['SvmOptimize'].append(val_Y)
            train_metrics['SvmOptimize'].append(fold_train_mae)
            valid_metrics['SvmOptimize'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'svm' in ALGOS_TO_RUN:
            svm_model = svm.SVR(kernel='rbf')
            svm_model.fit(pca_train_X, train_Y)
            pred_train_Y = svm_model.predict(pca_train_X)
            pred_valid_Y = svm_model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['svm'].append(train_Y)
            valid_Ys['svm'].append(val_Y)
            train_metrics['svm'].append(fold_train_mae)
            valid_metrics['svm'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'xgboost_optimize' in ALGOS_TO_RUN:
            model = xgb_parameter_optimize(pca_train_X, train_Y, XGB_PARAMETER_TYPE)
            pred_train_Y = model.predict(pca_train_X)
            pred_valid_Y = model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['xgboost_optimize'].append(train_Y)
            valid_Ys['xgboost_optimize'].append(val_Y)
            train_metrics['xgboost_optimize'].append(fold_train_mae)
            valid_metrics['xgboost_optimize'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)
            #models['xgboost_optimize'] = model

        if 'xgboost' in ALGOS_TO_RUN:
            # Xgboost
            model = XGBRegressor(
                learning_rate=0.01,  # 默认0.3
                n_estimators=500,  # 树的个数
                max_depth=5,
                # min_child_weight=1,
                # gamma=0,
                # subsample=0.8,
                # colsample_bytree=0.8,
                # scale_pos_weight=1
            )
            #print('pca_train_X', type(pca_train_X))
            #print('pca_train_X shape', pca_train_X.shape)
            #print('train_Y shape', train_Y.shape)
            model.fit(pca_train_X, train_Y)
            pred_train_Y = model.predict(pca_train_X)
            pred_valid_Y = model.predict(pca_val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['xgboost'].append(train_Y)
            valid_Ys['xgboost'].append(val_Y)
            train_metrics['xgboost'].append(fold_train_mae)
            valid_metrics['xgboost'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)
            #models['xgboost'] = model

        if 'stepwise' in ALGOS_TO_RUN:
            # stepwise forward selection (by p value)
            all_feature_indices = set(range(train_X.shape[1]))
            selected_feature_indices = [0, 1, 2, 3, 4, 5]

            def mp_get_pvalue(ind):
                """get p value for newly added feature, which is the last feature"""
                model = sm.OLS(train_Y, train_X[:, list(selected_feature_indices) + [ind]]).fit()
                pvalue = model.pvalues[-1]
                return pvalue

            def get_pvalue(Y, X):
                """get p value for newly added feature, which is the last feature"""
                model = sm.OLS(Y, X).fit()
                pvalue = model.pvalues[-1]
                return pvalue

            while len(selected_feature_indices) < MAX_N_FEATURE_SELECT:
                unselected_feature_indices = all_feature_indices - set(selected_feature_indices)
                unselected_feature_indice1_pvalue0 = 100  # some random large p-value
                selected_feature_index = 0  # some random index

                # multi-processing (doesn't seem to speed up, moreover, costs a lot more time)
                # import time
                # pool = multiprocessing.Pool(4)
                # unselected_feature_indices_list = list(unselected_feature_indices)
                # start_time = time.time()
                # unselected_feature_pvalues = pool.map(mp_get_pvalue, unselected_feature_indices_list)
                # print("takes {}...".format(time.time() - start_time))
                # selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
                # selected_feature_indices += [selected_feature_index]

                # construct array of pvalues
                unselected_feature_indices_list = list(unselected_feature_indices)
                unselected_feature_pvalues = [get_pvalue(train_Y, train_X[:, list(selected_feature_indices + [ind])])
                                              for ind in tqdm(unselected_feature_indices_list)]
                selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
                selected_feature_indices += [selected_feature_index]

            model = sm.OLS(train_Y, train_X[:, selected_feature_indices]).fit()
            pred_train_Y = model.predict(train_X[:, selected_feature_indices])
            pred_valid_Y = model.predict(val_X[:, selected_feature_indices])
            # print("avg Y: {}, train mae is: {}, val mae is: {}".format(np.mean(Y),
            #                                                            np.mean(abs(train_Y - pred_train_Y)),
            #                                                            np.mean(abs(val_Y - pred_valid_Y))))
            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['stepwise'].append(train_Y)
            valid_Ys['stepwise'].append(val_Y)
            train_metrics['stepwise'].append(fold_train_mae)
            valid_metrics['stepwise'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

        if 'knn' in ALGOS_TO_RUN:
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(train_X, train_Y)
            pred_train_Y = model.predict(train_X)
            factor = np.mean([train_Y[i] / pred_train_Y[i] for i in range(len(pred_train_Y))])
            pred_train_Y = factor * pred_train_Y
            pred_valid_Y = factor * model.predict(val_X)

            fold_train_mae = np.mean(abs(train_Y - pred_train_Y))
            fold_val_mae = np.mean(abs(val_Y - pred_valid_Y))
            train_Ys['knn'].append(train_Y)
            valid_Ys['knn'].append(val_Y)
            train_metrics['knn'].append(fold_train_mae)
            valid_metrics['knn'].append(fold_val_mae)
            train_ensemble1.append(train_Y - pred_train_Y)
            valid_ensemble1.append(val_Y - pred_valid_Y)

            #models['knn'] = model

        train_ensemble.append(np.mean(abs(np.sum(np.array(train_ensemble1), axis=0) / len(ALGOS_TO_RUN))))
        valid_ensemble.append(np.mean(abs(np.sum(np.array(valid_ensemble1), axis=0) / len(ALGOS_TO_RUN))))

    print(create_filename(OUTPUT_PATH, ALGOS_TO_RUN))
    for k in train_metrics:
        print("{}: avg train_Y: {}, avg val_Y: {}, train mae: {}, val mae: {}".format(k,
                                                                                      np.mean(train_Ys[k]),
                                                                                      np.mean(valid_Ys[k]),
                                                                                      np.mean(train_metrics[k]),
                                                                                      np.mean(valid_metrics[k])))
    print("ensemble: train mae: {}, val mae: {}".format(np.mean(train_ensemble),
                                                        np.mean(valid_ensemble)))
    algos = [k for k in train_metrics] + ['ensemble-{}'.format('-'.join(train_metrics.keys()))]
    algos_scores = [np.mean(valid_metrics[k]) for k in train_metrics] + [np.mean(valid_ensemble)]
    return algos[int(np.argmin(algos_scores))], algos_scores[int(np.argmin(algos_scores))]#, models[algos[int(np.argmin(algos_scores))]]

def predict(X, Y, test_X, best_algo):
    # PCA
    pred_test_Ys = []
    if 'xgboost' in best_algo or 'lasso'  or 'xgboost_optimize' or 'svm' in best_algo:
        #pca = PCA(n_components=40)
        #gpca.fit(X)
        #pca_X = pca_transform_measure(gpca, X)
        #pca_test_X = pca_transform_measure(test_X)
        #pca = PCA(n_components)
        pca_fit(pca, X)
        pca_X = pca_transform_measure(pca, X)
        pca_test_X = pca_transform_measure(pca, test_X)
        #print('pca_X shape:', pca_X.shape)
        #print('test_X shape:', pca_test_X.shape)
        #pca_val_X = pca_transform_measure(pca, val_X)
    if 'lasso' in best_algo:
        # lasso
        model = Lasso(alpha=1.0)
        model.fit(pca_X, Y)
        pred_test_Y = model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'fcn' in ALGOS_TO_RUN:
        model = build_fcn_model(n_components)
        model.fit(pca_X, Y.reshape(-1,1))
        pred_test_Y = model.predict(pca_test_X).ravel()
        pred_test_Ys.append(pred_test_Y)

    if 'svm' in ALGOS_TO_RUN:
        svm_model = svm.SVR(kernel='rbf')
        svm_model.fit(pca_X, Y)
        pred_test_Y = svm_model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'xgboost_optimize' in best_algo:
        # xgboost_optimize
        model = xgb_parameter_optimize(pca_X, Y, XGB_PARAMETER_TYPE)
        pred_test_Y = model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'xgboost' in best_algo:
        # Xgboost
        model = XGBRegressor(
            learning_rate=0.01,  # 默认0.3
            n_estimators=500,  # 树的个数
            max_depth=3,
            # min_child_weight=1,
            # gamma=0,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # scale_pos_weight=1
        )

        model.fit(pca_X, Y)
        pred_test_Y = model.predict(pca_test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'stepwise' in best_algo:
        # stepwise forward selection (by p value)
        all_feature_indices = set(range(X.shape[1]))
        selected_feature_indices = [0, 1, 2, 3, 4, 5]

        def mp_get_pvalue(ind):
            """get p value for newly added feature, which is the last feature"""
            model = sm.OLS(Y, X[:, list(selected_feature_indices) + [ind]]).fit()
            pvalue = model.pvalues[-1]
            return pvalue

        def get_pvalue(Y, X):
            """get p value for newly added feature, which is the last feature"""
            model = sm.OLS(Y, X).fit()
            pvalue = model.pvalues[-1]
            return pvalue

        while len(selected_feature_indices) < MAX_N_FEATURE_SELECT:
            unselected_feature_indices = all_feature_indices - set(selected_feature_indices)
            unselected_feature_indice1_pvalue0 = 100  # some random large p-value
            selected_feature_index = 0  # some random index

            # multi-processing (doesn't seem to speed up, moreover, costs a lot more time)
            # import time
            # pool = multiprocessing.Pool(4)
            # unselected_feature_indices_list = list(unselected_feature_indices)
            # start_time = time.time()
            # unselected_feature_pvalues = pool.map(mp_get_pvalue, unselected_feature_indices_list)
            # print("takes {}...".format(time.time() - start_time))
            # selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
            # selected_feature_indices += [selected_feature_index]

            # construct array of pvalues
            unselected_feature_indices_list = list(unselected_feature_indices)
            unselected_feature_pvalues = [get_pvalue(Y, X[:, list(selected_feature_indices + [ind])])
                                          for ind in tqdm(unselected_feature_indices_list)]
            selected_feature_index = unselected_feature_indices_list[int(np.argmin(unselected_feature_pvalues))]
            selected_feature_indices += [selected_feature_index]

        model = sm.OLS(Y, X[:, selected_feature_indices]).fit()
        pred_test_Y = model.predict(test_X[:, selected_feature_indices])
        pred_test_Ys.append(pred_test_Y)

    if 'knn' in best_algo:
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X, Y)
        pred_train_Y = model.predict(X)
        factor = np.mean([Y[i] / pred_train_Y[i] for i in range(len(pred_train_Y))])
        pred_test_Y = factor * model.predict(test_X)
        pred_test_Ys.append(pred_test_Y)

    if 'ensemble' in best_algo:
        n_algo = len(pred_test_Ys)
        pred_test_Y = np.sum(np.array(pred_test_Ys), axis=0) / n_algo

    return pred_test_Y


if __name__ == "__main__":
    # Processing data
    # hgt
    hgt_dataset, hgt_nc_attrs, hgt_nc_dims, hgt_nc_vars = run_ncdump(HGT_PATH, print_features=True)  # 845, 17, 73, 144
    # air
    air_dataset, air_nc_attrs, air_nc_dims, air_nc_vars = run_ncdump(AIR_PATH)  # 845, 17, 73, 144
    # slp
    #slp_dataset, slp_nc_attrs, slp_nc_dims, slp_nc_vars = run_ncdump(SLP_PATH)  # 845, 73, 144
    # rhum
    rhum_dataset, rhum_nc_attrs, rhum_nc_dims, rhum_nc_vars = run_ncdump(RHUM_PATH, print_features=True)  # 845, 73, 144
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
    #print(all_data)
    #print(type(all_data))
    models_city = {}
    best_mae = 0
    # train
    print("training model...")

    X = all_data['X']
    Y = all_data['Y']

    # deal with -9.xx e36 values
    X[X < -1000] = 0

    #print('X shape:', X.shape)
    best_algo, mae = train(X, Y)
    recipe = best_algo
    best_mae += mae
    create_pickle(recipe, create_filename(TRAIN_RECIPE_PATH, ALGOS_TO_RUN))
    #print(recipe)
    # predict
    output = {}
    # TODO: load recipe

    #pca_X = pca_transform_measure(gpca, X)
    #Y = knn_data[i]['Y']
    test_X = all_data['test_X']
    #test_X = pca_transform_measure(gpca, test_X)

    # deal with -9.xx e36 values
    X[X < -1000] = 0
    test_X[test_X < -1000] = 0

    print("testing model...")
    best_alg = recipe
    #print(models_city)
    #pca_test_X = pca.transform(test_X)
    pred_test_Y = predict(X, Y, test_X, best_algo)
    output = y_scaler.inverse_transform(pred_test_Y)
    create_pickle(output, create_filename(OUTPUT_PATH, ALGOS_TO_RUN))