"""
https://www.quora.com/What-is-the-best-way-to-combine-multiple-models-in-machine-learning-to-achieve-a-better-AUC-for-the-ROC-curve

One way to do this is to use a logistic regression to linearly fuse the systems.  This is a very simple fusion approach.
The targets for the logistic regression would be the binary class labels.
The fused score would be the weighted combination of system scores and an offset using the logistic regression parameters.

It also possible to use a nonlinear fusion system by training a neural network but this can be riskier in terms
of over fitting and generalizing to new data.

http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
An ensemble is a collection of models whose predictions are combined by weighted averaging or voting. Dietterich states that "A necessary and
sufficient condition for an ensemble of classifiers to be more accurate than any of its individual members is if the classfiers are accurate 
and diverse.
"""

import numpy as np
import pandas as pd

import load_data

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':

    # np.random.seed(0) # seed to shuffle the train set

    n_folds = 4
    n_threads = 2
    
	# verbose = False
    # shuffle = False

    print 'Loading train and test sets'
    trainset, train_label, testset = load_data.load()
    print 'Finished feature engineering for train and test sets'

    # if shuffle:
    #   idx = np.random.permutation(train_label.size)
    #   trainset = trainset[idx]
    #   train_label = train_label[idx]

    # Level 0 classifiers
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=n_threads, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=n_threads, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=n_threads, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=n_threads, criterion='entropy'),
            GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, subsample=0.5, max_depth=6)]

    # Stratified random shuffled cross validation
    skf = list(StratifiedKFold(train_label, n_folds, shuffle=True))

    print 'Creating train and test sets for blending'

    # Create train and test sets for blending and Pre-allocate the data
    blend_train = np.zeros((trainset.shape[0], len(clfs)))
    blend_test = np.zeros((testset.shape[0], len(clfs)))

    # For each classifier, we train the number of fold times (=len(skf))
    for clf_index, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (clf_index + 1)
        print clf

        blend_test_j = np.zeros((testset.shape[0], len(skf)))  # Number of testing data x Number of folds , we will take the mean of the predictions

        for fold_index, (train_index, valid_index) in enumerate(skf):
            print 'Fold [%s]' % (fold_index + 1)

            # Cross validation training and validation set
            X_train = trainset.iloc[train_index]
            y_train = train_label.iloc[train_index]
            X_valid = trainset.iloc[valid_index]
            y_valid = train_label.iloc[valid_index]

            clf.fit(X_train, y_train)

            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[valid_index, clf_index] = clf.predict_proba(X_valid)[:, 1]
            blend_test_j[:, fold_index] = clf.predict_proba(testset)[:, 1]

        # Take the mean of the predictions of the cross validation set. Each column is now a meta-feature
        blend_test[:, clf_index] = blend_test_j.mean(axis=1)
		
		# Another way of doing this instead of predicting on the cv set, the levell 0 estimator can be trained on the full data again
		# and take the prediction on the full testset
		#clf.fit(trainset, train_label)
		#blend_test[:, clf_index] = clf.predict_proba(testset)

    print
    print 'Blending using LogisticRegression'
    bclf = LogisticRegression()
    bclf.fit(blend_train, train_label)
    y_pred_proba = bclf.predict_proba(blend_test)[:, 1]

    print 'Linear stretch of predictions to [0,1]'
    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

    print "Writing Final Submission File"
    preds_out = pd.read_csv('data/sample_submission.csv')
    preds_out['QuoteConversion_Flag'] = y_pred_proba

    # preds_out = preds_out.set_index('QuoteNumber')
    preds_out.to_csv('homesite_blended_RF_ET_GBM_with_FE_nan.csv', index=False, float_format='%0.9f')

    print 'Done'
