import skorch
from pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data.dataset import Subset
from tripletnet.networks import TripletNetwork, lmelloEmbeddingNet, split_gridsearchparams
from tripletnet.classifiers.HierarchicalClassifier import HierarchicalClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV, KFold
from rpdbcs.utils.experiment import do_experiment, createPipeline
from tripletnet.datahandler import BalancedDataLoader
from torch.utils.data import DataLoader
from tripletnet.classifiers.MetricLearningEnsemble import MetricLearningEnsembleClassifier
from pytorch_metric_learning import losses
from adabelief_pytorch import AdaBelief
from itertools import combinations
from tripletnet.callbacks import TensorBoardCallback, TensorBoardEmbeddingCallback, ClassifierCallback, TensorBoardCallbackBase, ExtendedEpochScoring, createLoadEndState_callback
from datetime import datetime
import os
from rpdbcs.datahandler.dataset import readDataset
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Uncomment the two lines below if you need to ensure exact same results at multiple executions.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')
RANDOM_STATE = 0
if(RANDOM_STATE is not None):
    np.random.seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)


def _createBaseClassifier(build_grid_search=False):
    # return KNeighborsClassifier(3), {}
    clf = RandomForestClassifier(500, n_jobs=-1, max_features=8, random_state=RANDOM_STATE, min_impurity_decrease=1e-4)
    grid_params = {'max_features': [3, 5, 7]}

    if(build_grid_search):
        sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        return GridSearchCV(clf, grid_params, cv=sampler, scoring='f1_macro')
    return clf, grid_params


class VotingClassifierWrapper(VotingClassifier):
    def set_validation_dataset(self, X_val, y_val, validation_sampler):
        self.X_val = X_val
        self.y_val = y_val
        if(X_val is not None):
            for e in self.estimators:
                e[1].estimator['transformer'].set_validation_dataset(X_val, y_val)
                e[1].cv = validation_sampler


def scoreMetricNet(net, X, y):
    clf = QuadraticDiscriminantAnalysis(tol=1e-7)
    # clf = _createBaseClassifier(build_grid_search=True)
    Xtrain, Xvalid = X
    ytrain, yvalid = y

    X_emb = net.transform(Xtrain)
    clf.fit(X_emb, ytrain)
    yp = clf.predict(X_emb)
    score_train = f1_score(ytrain, yp, average='macro')

    if(Xvalid is not None):
        X_emb = net.transform(Xvalid)
        yp = clf.predict(X_emb)
        score_valid = f1_score(yvalid, yp, average='macro')
    else:
        score_valid = None
    return score_train, score_valid
    # net.history.record("train_f1-macro", score_train)
    # net.history.record(self.name + '_best', bool(is_best))

    # if(dataset_valid is not None):
    #     X_valid, y_valid = getData(dataset_valid)
    #     X_emb_valid = net.forward(X_valid).cpu().numpy()
    #     yp_valid = self.clf.predict(X_emb_valid)
    #     score_valid = f1_score(y_valid, yp_valid, average='macro')
    #     # self.writer.add_scalar("valid/f1_macro", score_valid, global_step=epoch)
    #     net.history.record("valid_f1-macro", score_valid)


def getcallbacks(monitor_loss):
    callbacks = [ExtendedEpochScoring(scoreMetricNet, lower_is_better=False,
                                      on_train=False, name='f1-macro', use_caching=False)]
    callbacks += createLoadEndState_callback(monitor_loss)

    return callbacks


def createMLE(loss_function_list, name, labels_name=None, ensemble_filter=0, ensemble_strategy='voting', **metricnet_params):
    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False,
                            'print_change_log': False}
    optimizer_parameters = {"optimizer__"+key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief
    parameters = {
        'device': 'cuda',
        'module': lmelloEmbeddingNet, 'module__num_outputs': 8,
        # 'init_random_state': 0,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'iterator_valid': DataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False,
        'margin_decay_delay': 0}
    parameters = {**parameters, **optimizer_parameters}

    metricnet_params, grid_search_params = split_gridsearchparams(metricnet_params)
    parameters.update(metricnet_params)

    parameters['criterion'] = TripletNetwork.MetricLearningLossWrapper
    parameters['criterion__miner'] = None

    params_list = []
    for lf in loss_function_list:
        p = dict(parameters)
        p['criterion__loss_func'] = lf
        subname = "%.5s" % lf.__class__.__name__

        dir_to_save = os.path.join("runs", CURRENT_TIME)
        swriter = TensorBoardCallbackBase.create_SummaryWriter(dir_to_save, name=name+' '+subname)
        callbacks = []
        # callbacks.append(ClassifierCallback())
        # callbacks.append(EpochScoring(scoreMetricNet, lower_is_better=False,
        #                               on_train=True, name='train f1-macro', use_caching=False))
        # callbacks.append(EpochScoring(scoreMetricNet, lower_is_better=False,
        #                               on_train=False, name='valid f1-macro', use_caching=False))
        # callbacks.append(TensorBoardEmbeddingCallback(swriter, labels_name=labels_name))
        callbacks.append(TensorBoardCallback(swriter, close_after_train=True))
        callbacks += getcallbacks("valid_loss")

        p['callbacks'] = callbacks

        params_list.append(p)

    base_clf, baseclf_params = _createBaseClassifier()
    tripletnets = []
    for i, params in enumerate(params_list):
        tripletnets.append(TripletNetwork(**params, init_random_state=i+10))
    # estimators = [("net%d" % i, createPipeline(T, base_clf, {}, baseclf_params))
    #               for i, T in enumerate(tripletnets)]
    estimators = [("net%d" % i, createPipeline(T, base_clf, {}, baseclf_params))
                  for i, T in enumerate(tripletnets)]
    clf = VotingClassifierWrapper(estimators=estimators, voting='soft')
    # clf = MetricLearningEnsembleClassifier(base_classifier=base_clf, ensemble_filter=ensemble_filter,
    #                                        ensemble_strategy=ensemble_strategy, metricnet_params=params_list,
    #                                        base_classif_param_grid=baseclf_params)

    return clf


def main(inputdata, outfile):
    params = {
        'device': 'cuda',
        'module': lmelloEmbeddingNet,
        'module__num_outputs': 8,
        'optimizer__lr': 1.0e-3,
        'max_epochs': 60,
        'batch_size': 100,
        'train_split': skorch.dataset.CVSplit(9, stratified=True),
        'cache_dir': '.myptcache',
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False, 'iterator_train__random_state': RANDOM_STATE,
        'iterator_valid': DataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False,
        # 'criterion': losses.ProxyAnchorLoss,
        'criterion': losses.TripletMarginLoss
    }
    # criterion_params = {
    #     'num_classes': 5,
    #     'embedding_size': params['module__num_outputs'],
    #     'margin': 0.2,
    #     'alpha': 128
    # }
    criterion_params = {
        'margin': 0.2,
        'triplets_per_anchor': 'all'
    }
    params.update({"criterion__"+p: v for p, v in criterion_params.items()})
    D = readDataset('%s/freq.csv' % inputdata, '%s/labels.csv' % inputdata,
                    remove_first=100, dtype=np.float32, discard_multilabel=False)
    D.normalize(37.28941975)
    _, labels_name = D.getMulticlassTargets()

    sampler = StratifiedKFold(n_splits=10, shuffle=False)
    # sampler = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=RANDOM_STATE)

    # loss_function_list = [losses.ProxyAnchorLoss(num_classes=5, embedding_size=8, margin=0.1, alpha=128),
    #                       losses.CosFaceLoss(num_classes=5, embedding_size=8, margin=0.1),
    #                       losses.ContrastiveLoss(neg_margin=0.1, pos_margin=0.0),
    #                       losses.GeneralizedLiftedStructureLoss(neg_margin=0.1, pos_margin=0.0),
    #                       losses.TripletMarginLoss(margin=0.1, triplets_per_anchor='all')
    #                       ]

    # ensemble = createMLE(list(loss_function_list)*2, name='ML-Ensemble', labels_name=labels_name, **params)
    # classifiers = [('ML-Ensemble', ensemble)]
    tripletnet = TripletNetwork(init_random_state=233,
                                callbacks=getcallbacks('f1-macro'), **params)
    tripletnet = createPipeline(tripletnet, _createBaseClassifier(build_grid_search=True), {}, {})
    classifiers = [('metricnet', tripletnet)]
    classifiers_ictai = [('RF', _createBaseClassifier(build_grid_search=True))]

    do_experiment(D, [], classifiers_ictai, sampler, outfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdata', type=str, required=True)
    parser.add_argument('-o', '--outfile', type=str, required=False)
    args = parser.parse_args()

    main(args.inputdata, args.outfile)
