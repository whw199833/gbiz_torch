# coding:utf-8
# @Author: Haowen Wang

import torch
import sklearn.metrics as sm


class AUC(object):
    """
        Wraps sklearn.metrics.roc_auc_score to calculate auc in gbiz_torch.
    """

    def __init__(self, average='macro'):
        self.average = average
        self.auc = sm.roc_auc_score

    def __call__(self, labels, pred):
        """
        Args:
            labels: array-like of shape (n_samples,) or (n_samples, n_classes)
            pred: array-like of shape (n_samples,) or (n_samples, n_classes)

        """
        return self.auc(labels, pred, average=self.average)


class Confusion_Matrix(object):
    """
        Wraps sklearn.metrics.confusion_matrix to calculate consution matrix in gbiz_torch.

        labels: array-like of shape (n_classes), default=None 
                List of labels to index the matrix. This may be used to reorder or select a subset of labels.
                If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
    """

    def __init__(self, labels=None):
        self.labels = labels
        self.confusion_matrix = sm.confusion_matrix

    def __call__(self, target, pred):
        """
        Args:
            target: array-like of shape (n_samples,) Ground truth (correct) target values.
            pred: array-like of shape (n_samples,) Estimated targets as returned by a classifier.

            labels : array-like of shape (n_classes), default=None
                    List of labels to index the matrix. This may be used to reorder

        output:
            Confusion_Matrix: shape: (num_classes, num_classes)

        """
        return self.confusion_matrix(pred, target, labels=self.labels)


class ACC_F1_score(object):
    """
        Wraps sklearn.metrics.f1_score to calculate acc and F1 in gbiz_torch.
    """

    def __init__(self):
        self.f1_score = sm.f1_score

    def __call__(self, labels, model_output):
        """
        Args:
            labels: 1d array-like, or label indicator array / sparse matrix
            model_output: 1d array-like, or label indicator array / sparse matrix
                          Estimated targets as returned by a classifier

        output:
            F1 cross average type output {'acc', 'Macro-F1', 'Weighted-F1'}

        """
        type_list = ['micro', 'macro', 'weighted']

        tmp_res = []
        for f_type in type_list:
            tmp_res.append(self.f1_score(labels, model_output, average=f_type))

        res = {'ACC': tmp_res[0], 'Macro-F1': tmp_res[1],
               'Weighted-F1': tmp_res[2]}

        return res


class Top_K_Acc(object):
    """
        Wraps sklearn.metrics.top_k_accuracy_score to calculate top K accuracy in gbiz_torch.
        k: top K probability classes has the right label class
    """

    def __init__(self, k=3):
        self.k = k
        self.top_k = sm.top_k_accuracy_score

    def __call__(self, labels, model_output):
        """
        Args:
            labels: array-like of shape (n_samples,)
            model_output: array-like of shape (n_samples,) or (n_samples, n_classes)

        output:
            value: top_k_accuracy_score

        """

        res = self.top_k(labels, model_output, k=self.k)

        return res


class Multi_Class_RP(object):
    """
        Wraps sklearn.metrics.precision_recall_fscore_support to cal per-label precisions, recalls, F1-scores and supports instead of averaging
        k: top K probability classes has the right label class
    """

    def __init__(self, average='micro'):
        self.average = average
        self.precision_recall_fscore_support = sm.precision_recall_fscore_support

    def __call__(self, labels, preds):
        """
        Args:
            labels: 1d array-like, or label indicator array / sparse matrix
                    Ground truth (correct) target values.
            model_output: 1d array-like, or label indicator array / sparse matrix
                          Estimated targets as returned by a classifier.

        output:
            precision (Precision score): float (if average is not None) or array of float, shape = [n_unique_labels]

            recall (Recall score): float (if average is not None) or array of float, shape = [n_unique_labels]

            fbeta_score (F-beta score): float (if average is not None) or array of float, shape = [n_unique_labels]

            support: None (if average is not None) or array of int, shape = [n_unique_labels]
                    The number of occurrences of each label in y_true.
        """
        res = self.precision_recall_fscore_support(labels, preds)
        return res
