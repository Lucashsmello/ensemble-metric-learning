from pytorch_metric_learning.losses import proxy_anchor_loss
from pytorch_metric_learning.losses.cosface_loss import CosFaceLoss
from rpdbcs.model_selection.GridSearchCV_norefit import GridSearchCV_norefit
import skorch
from sklearn.pipeline import Pipeline
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from rpdbcs.utils.experiment import do_experiment, createPipeline
from torch.utils.data import DataLoader
# from tripletnet.callbacks import TensorBoardCallback, TensorBoardEmbeddingCallback, TensorBoardCallbackBase, createLoadEndState_callback
# from tripletnet.datahandler import BalancedDataLoader
# from tripletnet.networks import TripletNetwork, lmelloEmbeddingNet, split_gridsearchparams
from pytorch_metric_learning import losses
from pytorch_metric_learning.reducers import MeanReducer
from adabelief_pytorch import AdaBelief
from itertools import combinations
from datetime import datetime
import os
from rpdbcs.datahandler.dataset import readDataset
from sklearn.ensemble import VotingClassifier

from skorch_extra.netbase import NeuralNetTransformer
from skorch_extra.callbacks import TensorBoardCallback, LoadEndState
from pytorch_balanced_sampler.sampler import BalancedDataLoader
from network import NetArchModel


CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')
RANDOM_STATE = 3
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def split_gridsearchparams(params):
    grid_search_params = {p: v for p, v in params.items() if isinstance(v, list) and 'list' not in p}
    params = {p: v for p, v in params.items() if not isinstance(v, list) or 'list' in p}

    return params, grid_search_params


def _createBaseClassifier(build_grid_search=False):
    # return KNeighborsClassifier(3), {}
    clf = RandomForestClassifier(500, n_jobs=-1, max_features=8, random_state=RANDOM_STATE)
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


def getCallbacks(save_dir_name):
    """
    Callbacks used by skorch. Four callbacks are used:
    - LoadEndState: At the end of training, load the best neuralnet of all epochs.
    - TensorBoardCallback: log results using tensorboard.
    """

    # log_dir = os.path.join("runs", CURRENT_TIME, save_dir_name)
    # swriter = SummaryWriter(log_dir=log_dir)

    from tempfile import mkdtemp

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=mkdtemp(),
                                                      monitor='valid_loss_best',
                                                      f_params='best_epoch_params.pt',
                                                      f_history=None, f_optimizer=None, f_criterion=None)
    loadbest_net_callback = LoadEndState(checkpoint_callback, delete_checkpoint=True)
    callbacks = [checkpoint_callback, loadbest_net_callback]
    # callbacks.append(TensorBoardCallback(swriter, close_after_train=True))

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
        'module': NetArchModel, 'module__num_outputs': 8,
        # 'init_random_state': 0,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'iterator_valid': DataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False}
    parameters = {**parameters, **optimizer_parameters}

    metricnet_params, _ = split_gridsearchparams(metricnet_params)
    parameters.update(metricnet_params)

    base_clf, baseclf_params = _createBaseClassifier()

    estimators = []
    for i, (lf, lf_params) in enumerate(loss_function_list):
        lf_params = {f'criterion__{k}': v for k, v in lf_params.items()}
        p = dict(parameters)
        p['criterion'] = lf
        subname = "%.5s" % lf.__name__

        callbacks = getCallbacks(name+' '+subname)
        p['callbacks'] = callbacks

        net = NeuralNetTransformer(**p, init_random_state=i+10)
        estimator = createPipeline(net, base_clf, lf_params, baseclf_params)
        estimators.append((f'net{i}', estimator))

    clf = VotingClassifierWrapper(estimators=estimators, voting='soft')
    # clf = MetricLearningEnsembleClassifier(base_classifier=base_clf, ensemble_filter=ensemble_filter,
    #                                        ensemble_strategy=ensemble_strategy, metricnet_params=params_list,
    #                                        base_classif_param_grid=baseclf_params)

    return clf


class FastNet(nn.Module):
    """
    Just used for debugging.
    """

    def __init__(self, num_outputs, num_inputs_channels=1):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(num_inputs_channels, 2, 5, padding=2), nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(10, stride=10),  # 6100 -> 610
            nn.Conv1d(2, 4, 5, padding=2), nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(10, stride=10),  # 610 -> 61
            nn.Flatten()
        )

        self.fc = nn.Sequential(nn.Linear(4 * 61, num_outputs))

    def forward(self, x):
        output = self.convnet(x)
        output = self.fc(output)
        return output


def main(inputdata, outfile):
    n_outs = 8
    params = {
        'device': 'cuda',
        'module': NetArchModel,
        'module__num_outputs': n_outs,
        'optimizer__lr': 1.0e-3,
        'max_epochs': 100,
        'batch_size': 100,
        'train_split': skorch.dataset.ValidSplit(9, stratified=True),
        'cache_dir': '.myptcache3',
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False, 'iterator_train__random_state': RANDOM_STATE,
        'iterator_valid': DataLoader, 'iterator_valid__num_workers': 0, 'iterator_valid__pin_memory': False,
    }
    n = 15

    D = readDataset('%s/freq.csv' % inputdata, '%s/labels.csv' % inputdata,
                    remove_first=100, dtype=np.float32, discard_multilabel=False)
    D.normalize(37.28941975)
    _, labels_name = D.getMulticlassTargets()

    sampler = StratifiedKFold(n_splits=10, shuffle=False)
    # sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

    # loss_function_list = [losses.ProxyAnchorLoss(num_classes=5, embedding_size=n_outs, margin=0.2, alpha=128),
    #                       losses.CosFaceLoss(num_classes=5, embedding_size=n_outs, margin=0.2),
    #                       losses.ContrastiveLoss(neg_margin=0.2),
    #                       losses.GeneralizedLiftedStructureLoss(neg_margin=0.2, pos_margin=0.0),
    #                       losses.TripletMarginLoss(margin=0.2, triplets_per_anchor='all', reducer=MeanReducer())
    #                       ]

    proxyanchor_params = {'num_classes': [5], 'embedding_size': [n_outs],
                          'margin': [0.25, 0.5, 1.0], 'alpha': [32, 64, 128]}
    cosface_params = {'num_classes': [5], 'embedding_size': [n_outs],
                      'margin': [0.25, 0.5, 1.0], 'scale': [32, 64, 128]}
    contras_params = {'neg_margin': [0.25, 0.5, 1.0], 'pos_margin': [0.0, 0.05, 0.1], 'reducer': [MeanReducer()]}
    lsl_params = {'neg_margin': [0.25, 0.5, 1.0], 'pos_margin': [0.0, 0.05, 0.1]}
    triplet_params = {'margin': [0.25, 0.5, 1.0], 'triplets_per_anchor': [1, 5, 25], 'reducer': [MeanReducer()]}

    loss_function_list = [(losses.ProxyAnchorLoss, proxyanchor_params),
                          (losses.CosFaceLoss, cosface_params),
                          (losses.ContrastiveLoss, contras_params),
                          (losses.GeneralizedLiftedStructureLoss, lsl_params),
                          (losses.TripletMarginLoss, triplet_params)
                          ]

    classifiers = []
    for i in range(1, len(loss_function_list)+1):
        m = n % i
        d = n//i
        for c in combinations(loss_function_list, i):
            name = ".".join(["%.3s" % l.__name__ for l, _ in c])
            name = 'ML-Ensemble%d_%s' % (i, name)
            C = list(c)*d
            if(m > 0):
                C += (list(c)[:m])
            assert(len(C) == n)
            classifiers.append((name, createMLE(C, name=name, labels_name=labels_name, **params)))
    classifiers_ictai = [('RF', _createBaseClassifier(build_grid_search=True))]

    do_experiment(D, classifiers, classifiers_ictai, sampler, outfile)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdata', type=str, required=True)
    parser.add_argument('-o', '--outfile', type=str, required=False)
    args = parser.parse_args()

    main(args.inputdata, args.outfile)
