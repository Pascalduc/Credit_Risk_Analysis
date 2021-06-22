# Credit Risk Analysis

## Overview of Analysis

The purpose of this analysis was to use different machine learning models from the `scikit-learn` library to find the best one at predicting credit card risk. 

### Results

•	At first, we imported a credit card dataset from `LendingClub` using `pandas.read_csv` and transformed the data for our usage. Empty columns and rows were removed with `dropna` function, issued loans were also removed by filtering, interest rates were converted to numerical values, and the loans were classified into `low risk` and `high risk` based on their status.

•	In the next step, we assigned the X variables with the `get_dummies()` method to transform string categories into numerical values. Then we selected the `loan_status` column for our target values and counted the classes. 

![loan_status](Resources/loan_status.png)

•	As we can see, only 347 loans out of 68,817 are high risk, representing 0.5% of the analyzed data. 

•	We then sampled our training and testing data and proceed to fitting and testing our prediction models.

#### RandomOverSampler

![RandomOversampler](Resources/RandomOversampler.png)


#### SMOTE Oversampling

![SMOTE](Resources/SMOTE.png)


#### ClusterCentroids Undersampling

![ClusterCentroids](Resources/ClusterCentroids.png)


#### SMOTEENN Combination

![SMOTEENN](Resources/SMOTEENN.png)


#### BalancedRandomForestClassifier

![BalancedRandomForestClassifier](Resources/BalancedRandomForestClassifier.png)


#### EasyEnsembleClassifier

![EasyEnsembleClassifier](Resources/EasyEnsembleClassifier.png)


### Summary

•	In summary, we can see that the best balanced accuracy score is achieve with the `EasyEnsembleClassifier` model at 93.2%. The second best was the `BalancedRandomForestClassifier` with an accuracy score of 78.9%, while the oversampling and combination models were in the ~60-69%. The worst performing model is the undersampling `ClusterCentroids` at 54.5%.

•	When looking at the precision values for all models, they all had excellent overall and low-risk scores, but poor high-risk loans scores. This is due to high number of low-risk loans and very low number of high-risk loans skewing the data. Therefore, the high number of false positive for the high-risk loans, is not affecting the overall picture a lot since that risk category represents only ~0.5% of the total loans. `TP/(TP + FP)`

•	The recall or sensitivity is looking at the true positivity rate `TP/(TP + FN)`. Again, the model with the best sensitivity for both high- and low-risk loans was the `EasyEnsembleClassifier` with 92% and 94% respectively. 

•	In our case, precision is not as important as sensitivity since we want to correctly predict all or most of the bad loans. The `EasyEnsembleClassifier` is falsely flagging 983 loans as high-risk but correctly predict 93 out of 101. With this model, the bank would decline (93+983)/(93+983+8+16121) = 6.3% of loan applications to catch 92% of the 0.5% high-risk loans. Since each default loan is a huge loss for the bank, I would recommend the `EasyEnsembleClassifier` model, unless high interest from falsely flagged and declined loans would be enough to cover for these losses.

