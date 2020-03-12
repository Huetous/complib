import numpy as np
import inspect

# ---------------------------------------------------------------------------
class ClassificationMetrics:
    def __init__(self, cm):
        print(cm)
        tn, fp, fn, tp = cm.ravel()
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp

        self.N0 = tn + fp
        self.N1 = fn + tp

        self.n0 = fn + tn
        self.n1 = fp + tp

    def info(self):
        print('Variables:')
        for v in self.__dict__:
            print(f'{v} = {self.__dict__[v]}')

    def full(self):
        print('FPR:', self.fpr())
        print('TNR:', self.tnr())
        print('FNR:', self.fnr())
        print('TPR:', self.tpr())
        print('-' * 5)
        print('TNR + FPR = 1 is', self.fpr() + self.tnr() == 1)
        print('FNR + TPR = 1 is', self.fnr() + self.tpr() == 1)
        print('-' * 5)
        print('NPV:', self.npv())
        print('PPV:', self.ppv())
        print('FDR:', self.fdr())
        print('PPV + FDR = 1 is', self.ppv() + self.fdr() == 1)
        print('-' * 5)
        print('Accuracy: ', self.acc())
        print('F-score:', self.f_score())
        print('MCC:', self.mcc())

    # False Positive Rate | Type I error
    # Fraction of 0->1
    def fpr(self):
        return self.fp / self.N0

    # False Negarive Rate | Type II error
    # Fraction of 1->0
    def fnr(self):
        return self.fn / self.N1

    # True Negative Rate | Specificity
    # TNR = 1 - FPR
    # Fraction of 0->0
    def tnr(self):
        return self.tn / self.N0

    # True Positive Rate | Recall , Sensitivity
    # Fraction of 1->1 in N1
    def tpr(self):
        return self.tp / self.N1

    # Negative Predictive Value
    # Precision for a negative class
    # Fraction of 0->0 in n0
    def npv(self):
        return self.tn / self.n0

    # Positive Predictive Value | Precision
    # Fraction of 1->1 in n1
    def ppv(self):
        return self.tp / self.n1

    # False Discovery Rate
    # 1 - precision
    # Fraction of 0->1 in n1
    def fdr(self):
        return self.fp / self.n1

    # Accuracy
    def acc(self):
        return (self.tp + self.tn) / (self.N0 + self.N1)

    # larger beta -> more attention to recall
    def f_score(self, b=1):
        prec = self.ppv()
        rec = self.tpr()
        b_sq = b ** 2
        return (1 + b_sq) * prec * rec / (b_sq * prec + rec)

    # Matthews Correlation Coefficient
    # between pred classes and ground truth
    def mcc(self):
        a = self.tp + self.fp
        b = self.tp + self.fn
        c = self.tn + self.fp
        d = self.tn + self.fn
        return (self.tp * self.tn - self.fp * self.fn) / np.sqrt(a * b * c * d)
