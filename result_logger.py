import os

import h5py
import numpy as np
import xlwt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class ResultLogger:

    def __init__(self, tag, logdir='.', verbose=False):

        super().__init__()
        self.tag = tag
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        self.verbose = verbose
        self.class_num = 0
        self.results = []
        # training metric
        self.training_loss = []
        self.training_accuracies = []
        self.training_time = []
        # test metric
        self.acs_accuracies = []  # Average Class Specific Accuracy
        self.precisions = []
        self.recalls = []
        self.f_macros = []
        self.f_micros = []
        self.g_macros = []
        self.g_micros = []
        self.specificities = []
        self.accuracies_per_class = []
        self.g_per_class = []
        self.reports = []
        self.test_time = []

    def add_test_metrics(self, y_true, y_pred, time=0.):
        self.test_time.append(time)
        y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)
        self.class_num = max(self.class_num, len(np.unique(y_true)))
        report = classification_report(y_true, y_pred, digits=5, output_dict=True)
        self.reports.append(report)

        cnf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(self.class_num)))
        if self.verbose:
            print(cnf_matrix)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        cs_accuracy = TP / cnf_matrix.sum(axis=1)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        specificity = TN / (FP + TN)
        self.accuracies_per_class.append(cs_accuracy)
        self.acs_accuracies.append(cs_accuracy.mean())
        self.precisions.append(precision.mean())
        self.recalls.append(recall.mean())
        self.specificities.append(specificity.mean())

        f1_macro = (2 * precision * recall / (precision + recall)).mean()
        f1_micro = 2 * TP.sum() / (2 * TP.sum() + FP.sum() + FN.sum())
        self.f_macros.append(f1_macro)
        self.f_micros.append(f1_micro)

        g_marco = ((recall * specificity) ** 0.5).mean()
        g_micro = ((TP.sum() / (TP.sum() + FN.sum())) * (TN.sum() / (TN.sum() + FP.sum()))) ** 0.5
        self.g_macros.append(g_marco)
        self.g_micros.append(g_micro)
        self.g_per_class.append((recall * specificity) ** 0.5)

    def add_training_metrics(self, loss, accuracy, time=0.):
        self.training_loss.append(loss)
        self.training_accuracies.append(accuracy)
        self.training_time.append(time)

    def save_latent_code(self, epoch, mu, logsigma, z, y):
        mu = mu.astype(dtype=np.float32)
        logsigma = logsigma.astype(dtype=np.float32)
        z = z.astype(dtype=np.float32)
        y = y.astype(dtype=np.float32)
        filename = self.tag + "_latent_%05d.hdf5" % epoch
        with h5py.File(self.logdir + os.sep + filename, "w") as f:
            f.create_dataset("mu", data=mu)
            f.create_dataset("logsigma", data=logsigma)
            f.create_dataset("z", data=z)
            f.create_dataset("y", data=y)
            f.attrs['epoch'] = epoch

    def save_latent_code_new(self, epoch, z, y, mu, logsigma, theta_p, u_p, lambda_p):
        mu = mu.astype(dtype=np.float32)
        logsigma = logsigma.astype(dtype=np.float32)
        z = z.astype(dtype=np.float32)
        y = y.astype(dtype=np.float32)
        theta_p = theta_p.astype(dtype=np.float32)
        u_p = u_p.astype(dtype=np.float32)
        lambda_p = lambda_p.astype(dtype=np.float32)
        filename = self.tag + "_latent_%05d.hdf5" % epoch
        with h5py.File(self.logdir + os.sep + filename, "w") as f:
            f.create_dataset("mu", data=mu)
            f.create_dataset("logsigma", data=logsigma)
            f.create_dataset("z", data=z)
            f.create_dataset("y", data=y)
            f.create_dataset("theta_p", data=theta_p)
            f.create_dataset("u_p", data=u_p)
            f.create_dataset("lambda_p", data=lambda_p)
            f.attrs['epoch'] = epoch

    def save_prediction(self, epoch, labels, predicts, probs, time=0.):
        labels = labels.astype(dtype=np.int8)
        predicts = predicts.astype(dtype=np.int8)
        self.add_test_metrics(labels, predicts, time)
        probs = probs.astype(dtype=np.float32)
        filename = self.tag + "_proba_%05d.hdf5" % epoch
        with h5py.File(self.logdir + os.sep + filename, "w") as f:
            f.create_dataset("label", data=labels)
            f.create_dataset("predict", data=predicts)
            f.create_dataset("probability", data=probs)
            f.attrs['epoch'] = epoch

    def save_metrics(self):
        # save evaluation results
        workbook = xlwt.Workbook()
        sheet1 = workbook.add_sheet('evaluation_metrics')
        sheet2 = workbook.add_sheet('evaluation_metric_per_class')
        sheet3 = workbook.add_sheet('training_metrics')
        titles1 = ['rec_ma', 'pre_ma', 'spe_ma', 'acsa', 'f_ma', 'f_mi', 'g_ma', 'g_mi', 'time']
        for i, title in enumerate(titles1):
            sheet1.write(0, i, title)
        for i in range(len(self.acs_accuracies)):
            row = i + 1
            sheet1.write(row, 0, self.recalls[i])
            sheet1.write(row, 1, self.precisions[i])
            sheet1.write(row, 2, self.specificities[i])
            sheet1.write(row, 3, self.acs_accuracies[i])
            sheet1.write(row, 4, self.f_macros[i])
            sheet1.write(row, 5, self.f_micros[i])
            sheet1.write(row, 6, self.g_macros[i])
            sheet1.write(row, 7, self.g_micros[i])
            sheet1.write(row, 8, self.test_time[i])

        row = 0
        for i in range(len(self.acs_accuracies)):
            titles2 = ['epoch ' + str(i), 'accuracy', 'precision', 'recall', 'f1-score', 'g-score', 'support']
            for j, title in enumerate(titles2):
                sheet2.write(row, j, title)
            for j in range(self.class_num):
                row += 1
                sheet2.write(row, 0, 'class ' + str(j))
                if str(j) in self.reports[i] and j < len(self.accuracies_per_class[i]):
                    sheet2.write(row, 1, self.accuracies_per_class[i][j])
                    sheet2.write(row, 2, self.reports[i][str(j)]['precision'])
                    sheet2.write(row, 3, self.reports[i][str(j)]['recall'])
                    sheet2.write(row, 4, self.reports[i][str(j)]['f1-score'])
                    sheet2.write(row, 5, self.g_per_class[i][j])
                    sheet2.write(row, 6, self.reports[i][str(j)]['support'])
            row += 2

        titles3 = ['loss', 'accuracy', 'time']
        for i, title in enumerate(titles3):
            sheet3.write(0, i, title)
        for i in range(len(self.training_loss)):
            row = i + 1
            sheet3.write(row, 0, self.training_loss[i])
            sheet3.write(row, 1, self.training_accuracies[i])
            sheet3.write(row, 2, self.training_time[i])

        filename = self.tag + '_result' + '.xls'
        workbook.save(self.logdir + os.sep + filename)

    def reset(self):
        self.class_num = 0
        self.results.clear()
        self.acs_accuracies.clear()
        self.precisions.clear()
        self.recalls.clear()
        self.f_macros.clear()
        self.f_micros.clear()
        self.g_macros.clear()
        self.g_micros.clear()
        self.specificities.clear()
        self.accuracies_per_class.clear()
        self.g_per_class.clear()
        self.reports.clear()
