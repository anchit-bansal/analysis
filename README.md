# analysis
Internship assignment for RedCarpetUp Incorporation

# Contents
 R.O.C.K.S submission.ipynb (Code file)
 2010 Federal STEM Education Inventory Data Set.csv (Data)
 ROCKS_submission.py(Pytest implementation)

# Data
The data is downloaded from the link provided by the company.

# Ipython Notebook ( R.O.C.K.S submission)
Cell 1: Required imports

Cell 2: Reading file

Cell 3: Forward filling of column headers for columns having common header

# Step 1

Cell 4: Creating Dataframe of required columns (FY2008 and FY2009)

Cell 5: Data cleaning

Cell 6: Display df

Cell 7: Computing growth percentage and creating label column

Cell 8: Step 1 output

# Step 2

Cell 9: Univariate distribution of all non-funding columns except index. I did bit mapping here to get all columns in 1/0 format. Plotting the columns against the attribute probability.

Cell 10: Computing mutual_info_score for all non_funding_columns against the label column and creating a new dataframe for same.

# Step 3

Cell 11: Dividing data in 7:3 for train:test.

Cell 12: Removing characters like ([,],<) from columns headers because of requirement in xgboost library.

Cell 13: Display label_train.

Cell 14: Creating xgboost model and training it with the training data.

Cell 15: Predicting label for test data.

Cell 16: Feature selection for maximum roc_auc_score from all the thresholds in model.feature_importances_. Feature selection changes with each train sample. So running again might give varying results.

Cell 17: Printing the best score obtained.

Cell 18: Printing number of features selected.

Cell 19: Printing features selected.

# ROCKS_submission.py (Pytest implementation)
Included pytest test cases on explicitely defined functions.

Test Cases:

1) Checks if the get_test_size() function return exact 30% of data size.

2) Checks if test data is smaller than training data for function get_thresh().

3) Checks if required model type is passed in function get_thresh().

4) Checks if label dataframe passed in function get_thresh() is single column dataframe.

