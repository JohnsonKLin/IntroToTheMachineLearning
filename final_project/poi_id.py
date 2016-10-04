#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'director_fees','loan_advances', 'bonus', 'other','restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
# You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#remove negative value
for key in data_dict.keys():
    for feature in data_dict[key].keys():
        if data_dict[key][feature] < 0 :
            data_dict[key][feature] *= -1

print 'Total number of datapoint: {}'.format(len(data_dict.keys()))
print 'Allocation across classes of POI and non-POI: {},{}'.format(len([1 for key in  data_dict.keys() if data_dict[key]['poi'] == 1]),len([0 for key in  data_dict.keys() if data_dict[key]['poi'] == 0]))
print 'Total Number of features : {}'.format(len(features_list))
print 'Number of features in given dataset: {}'.format(len(data_dict['METTS MARK'].keys()))
### Task 2: Remove outliers
my_dataset = data_dict
my_dataset.pop("TOTAL",0)
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK",0)
my_dataset.pop("LOCKHART EUGENE E", 0)
feature_na_value= {}
for key in my_dataset.keys():
    for feature_i in my_dataset[key].keys():
        if my_dataset[key][feature_i] == 'NaN':
            if feature_i in feature_na_value:
                feature_na_value[feature_i] += 1
            else:
                feature_na_value[feature_i] = 1

print "Mising value for the feature"
for missing_value in feature_na_value.keys():
    print "{}:\t{}".format(missing_value,feature_na_value[missing_value])


def ratio_two_variable(numerator,denumerator,output_features_name):
    if my_dataset[key][numerator]== 'NaN' or my_dataset[key][denumerator] == 'NaN' or my_dataset[key][denumerator] == 0:
        my_dataset[key][output_features_name]=0
    else:
        my_dataset[key][output_features_name]= float(my_dataset[key][numerator])/float(my_dataset[key][denumerator])
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in my_dataset.keys():
    ratio_two_variable('from_poi_to_this_person','from_messages','from_poi_to_this_person_to_from_message_ratio')
    ratio_two_variable('shared_receipt_with_poi','from_messages','shared_receipt_with_poi_to_from_message_ratio')
    ratio_two_variable('from_this_person_to_poi','to_messages','from_this_person_to_poi_to_to_message_ratio')
    ratio_two_variable('long_term_incentive','total_payments','long_term_to_total_payments_ratio')
    ratio_two_variable('salary','total_payments','salary_to_total_payments_ratio')
    ratio_two_variable('bonus','total_payments','bonus_to_total_payments_ratio')
    ratio_two_variable('expenses','total_payments','expenses_to_total_payments_ratio')
    ratio_two_variable('other','total_payments','other_to_total_payments_ratio')
    ratio_two_variable('long_term_incentive','total_payments','long_to_total_payments_ratio')
    ratio_two_variable('deferred_income','total_payments','deferall_income_to_total_payments_ratio')
    ratio_two_variable('exercised_stock_options','total_stock_value','exe_to_total_stock')
    ratio_two_variable('restricted_stock','total_stock_value','res_to_total_stock')
    ratio_two_variable('restricted_stock_deferred','total_stock_value','res_deff_to_total_stock')
    ratio_two_variable('total_payments','total_stock_value','total_payments_to_total_stock_value_ratio')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
gau_clf = GaussianNB()

from sklearn import tree
tree_clf= tree.DecisionTreeClassifier(min_samples_split=2)
parameters = {'min_samples_split': [1,2,5,10]}

from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=500)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()

from sklearn.linear_model import SGDClassifier
g_clf = SGDClassifier(loss='log')

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
lgr_clf = LogisticRegression(C=10000000, class_weight='balanced',dual=False,fit_intercept=True, intercept_scaling=1, max_iter=100,multi_class='ovr', penalty='l2', random_state=None,solver='liblinear', tol=1e-05, verbose=0 )
lgr_parameters = {"C":[1,5,10,1000,10000,10000000],'class_weight':['balanced'],'dual':[False],"fit_intercept":[True],'intercept_scaling':[1],'max_iter':[100],'multi_class':['ovr'],'penalty':['l2'],'random_state':[None],'solver':['liblinear'],'tol':[1e-05],'verbose':[0]}
gird_lgr_clf = GridSearchCV(lgr_clf,lgr_parameters)
gird_clf = GridSearchCV(tree_clf,parameters)

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(random_state=42)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

pip_clf = Pipeline([("minmax", MinMaxScaler()),("reduce_feature", PCA(n_components=18)), ("clf", lgr_clf)])

clf = pip_clf
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
