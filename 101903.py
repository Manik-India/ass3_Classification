#importing necessary libraries
import pandas as pd
import pycaret
import sys
import os

#reading arguments that we have passed in terminal
input = sys.argv[1]
df = pd.read_csv(input)  #read csv file
from pycaret.classification import *

#Question 1: Accuracy without/ with Normalization
s = setup(data=df, target='Target', silent=True)
cm = compare_models()
cm = pull()
new_df=pd.DataFrame()
new_df['Model'] = cm['Model']
new_df['Accuracy without Normalization'] = cm['Accuracy']

setup(data=df, target='Target', normalize = True, normalize_method = 'zscore', silent=True)
cm = compare_models()
cm = pull()
new_df['Accuracy with zscore'] = cm['Accuracy']

setup(data=df, target='Target', normalize = True, normalize_method = 'minmax', silent=True)
cm = compare_models()
cm = pull()
new_df['Accuracy with minmax'] = cm['Accuracy']

setup(data=df, target='Target', normalize = True, normalize_method = 'maxabs', silent=True)
cm = compare_models()
cm = pull()
new_df['Accuracy with maxabs'] = cm['Accuracy']

setup(data=df, target='Target', normalize = True, normalize_method = 'robust', silent=True)
cm = compare_models()
cm = pull()
new_df['Accuracy with robust'] = cm['Accuracy']

new_df.to_csv('output-101903077-Normalization.csv')

#Question 2: Accuracy without/ with Feature Selection
new_df1=pd.DataFrame()
s = setup(data=df, target='Target', silent=True)
cm = compare_models()
cm = pull()
new_df1['Model'] = cm['Model']
new_df1['Accuracy without Feature Selection'] = cm['Accuracy']

setup(data=df, target='Target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.2, silent=True)
cm = compare_models()
cm = pull()
new_df1['Accuracy with classic = 0.2'] = cm['Accuracy']

setup(data=df, target='Target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.5, silent=True)
cm = compare_models()
cm = pull()
new_df1['Accuracy with classic = 0.5'] = cm['Accuracy']

setup(data=df, target='Target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.2, silent=True)
cm = compare_models()
cm = pull()
new_df1['Accuracy with boruta = 0.2'] = cm['Accuracy']

setup(data=df, target='Target', feature_selection = True, feature_selection_method = 'boruta', feature_selection_threshold = 0.5, silent=True)
cm = compare_models()
cm = pull()
new_df1['Accuracy with boruta = 0.5'] = cm['Accuracy']

new_df1.to_csv('output-101903077-FeatureSelection.csv')

#Question 3: Accuracy without/ with Outlier Removal
new_df2=pd.DataFrame()
s = setup(data=df, target='Target', silent=True)
cm = compare_models()
cm = pull()
new_df2['Model'] = cm['Model']
new_df2['Accuracy without Outlier Removal'] = cm['Accuracy']

setup(data=df, target='Target', remove_outliers = True, outliers_threshold = 0.02, silent=True)
cm = compare_models()
cm = pull()
new_df2['Accuracy with Threshold = 0.02'] = cm['Accuracy']

setup(data=df, target='Target', remove_outliers = True, outliers_threshold = 0.04, silent=True)
cm = compare_models()
cm = pull()
new_df2['Accuracy with Threshold = 0.04'] = cm['Accuracy']

setup(data=df, target='Target', remove_outliers = True, outliers_threshold = 0.06, silent=True)
cm = compare_models()
cm = pull()
new_df2['Accuracy with Threshold = 0.06'] = cm['Accuracy']

setup(data=df, target='Target', remove_outliers = True, outliers_threshold = 0.08, silent=True)
cm = compare_models()
cm = pull()
new_df2['Accuracy with Threshold = 0.08'] = cm['Accuracy']

new_df2.to_csv('output-101903077-OutlierRemoval.csv')

#Question 4: Accuracy without/ with PCA
new_df3=pd.DataFrame()
s = setup(data=df, target='Target', silent=True)
cm = compare_models()
cm = pull()
new_df3['Model'] = cm['Model']
new_df3['Accuracy without PCA'] = cm['Accuracy']

setup(data=df, target='Target', pca = True, pca_method = 'linear', silent=True)
cm = compare_models()
cm = pull()
new_df3['Accuracy with Method = linear'] = cm['Accuracy']

setup(data=df, target='Target', pca = True, pca_method = 'kernel', silent=True)
cm = compare_models()
cm = pull()
new_df3['Accuracy with Method = kernel'] = cm['Accuracy']

setup(data=df, target='Target', pca = True, pca_method = 'incremental', silent=True)
cm = compare_models()
cm = pull()
new_df3['Accuracy with Method = incremental'] = cm['Accuracy']

new_df3.to_csv('output-101903077-PCA.csv')

#Comparing models according to maximum accuracy
setup(data=df, target='Target', feature_selection = True, feature_selection_method = 'classic', feature_selection_threshold = 0.2, normalize = True, normalize_method = 'minmax', silent=True)
cm = compare_models()

#Creating Best Model accoding to gbc as, gbc and random forest get same and maximum accuracy from above comparing models
gbcModel = create_model('gbc')

#Question 5: Plotting Confusion Matrix according to Best Model
plot_model(gbcModel, plot='confusion_matrix', save = True)
os.rename('Confusion Matrix.png','output-101903077-ConfusionMatrix.png')

#Question 6: Plotting Learning Curve according to Best Model
plot_model(gbcModel, plot='learning', save = True)
os.rename('Learning Curve.png','output-101903077-LearningCurve.png')

#Question 7: Plotting AUC(Area Under Curve) according to Best Model
plot_model(gbcModel, plot='auc', save = True)
os.rename('AUC.png','output-101903077-AUC.png')

#Question 8: Plotting Decision Boundary according to Best Model
plot_model(gbcModel, plot='boundary', save = True)
os.rename('Decision Boundary.png','output-101903077-DecisionBoundary.png')

#Question 9: Plotting Feature Selection according to Best Model
gbcModel = create_model('gbc', verbose=False)
plot_model(gbcModel, plot='feature', save=True)
os.rename('Feature Importance.png','output-101903077-FeatureImportance.png')