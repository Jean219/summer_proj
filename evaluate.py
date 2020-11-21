import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


preds = pd.read_csv("./preds.csv",sep=r'\s*,\s*')
pred_probs = []

for _, row in preds.iterrows():
    pred_prob = [row['p_false'],row['p_true']]
    pred_probs.append(pred_prob)
# print(pred_probs)

# print(preds.head)
# print(preds.loc[:,['y_pred','y_true']])
# print(len(pred_probs))

print('confusion_matrix:')
print(metrics.confusion_matrix(preds['y_true'], preds['y_pred']))

print('log_loss:',metrics.log_loss(preds['p_label'],pred_probs))
print('accuracy_score:',metrics.accuracy_score(preds['y_true'], preds['y_pred']))
print('f1_score:',metrics.f1_score(preds['y_true'], preds['y_pred'], average='macro'))
print('precision_score:',metrics.precision_score(preds['y_true'], preds['y_pred'], average='macro'))
print('recall_score:',metrics.recall_score(preds['y_true'], preds['y_pred'], average='macro'))

fpr,tpr,thresholds = metrics.roc_curve(preds['y_true'],preds['y_pred'],pos_label=2)
roc_auc = metrics.auc(fpr, tpr)  # calculate AUC
print('roc_auc:',roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # false positive: x-axis, true positive: y-axis
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()