# CustomerReviewAnalysis

Train w/ 1 Page (7 Review):

Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1 Score: 1.0

Train w/ 10 Pages (70 Review):

Accuracy: 0.8571428571428571
Precision: 0.7346938775510203
Recall: 0.8571428571428571
F1 Score: 0.7912087912087912

2nd try:

Accuracy: 0.14285714285714285
Precision: 0.07142857142857142
Recall: 0.14285714285714285
F1 Score: 0.09523809523809523

Increase in Weight Decay to 0.01 to 0.05

[tensor(5), tensor(5), tensor(4), tensor(5), tensor(3), tensor(4), tensor(5), tensor(3), tensor(5), tensor(5), tensor(4), tensor(5), tensor(5), tensor(5)]
[5 5 5 5 5 5 5 5 5 5 5 5 5 5]
C:\Users\UNIDOCS\miniconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Accuracy: 0.6428571428571429
Precision: 0.41326530612244905
Recall: 0.6428571428571429
F1 Score: 0.5031055900621119

Train w/ 20 Pages (140 Reviews):

Accuracy: 0.6785714285714286
Precision: 1.0
Recall: 0.6785714285714286
F1 Score: 0.8001930501930502

Increase in Weight Decay to 0.01 to 0.05

[tensor(5), tensor(3), tensor(5), tensor(5), tensor(4), tensor(5), tensor(5), tensor(3), tensor(4), tensor(3), tensor(5), tensor(4), tensor(5), tensor(4), tensor(5), tensor(3), tensor(5), tensor(5), tensor(3), tensor(5), tensor(4), tensor(3), tensor(5), tensor(5), tensor(5), tensor(4), tensor(5), tensor(5)]
[5 2 5 5 5 5 5 2 5 2 5 5 5 5 5 2 5 5 2 5 5 2 5 5 5 5 5 5]
C:\Users\UNIDOCS\miniconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
C:\Users\UNIDOCS\miniconda3\Lib\site-packages\sklearn\metrics\_classification.py:1509: UndefinedMetricWarning: Recall is ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Accuracy: 0.5714285714285714
Precision: 0.4155844155844156
Recall: 0.5714285714285714
F1 Score: 0.48120300751879697