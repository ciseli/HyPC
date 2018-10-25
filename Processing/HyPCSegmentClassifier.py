"""
HyPC segment classifier

Description:
    Classify segments of a HyPC point cloud

Required inputs:
    - HyPC point cloud file

Created by: Christopher Iseli
Last Modified: 20/10/2018 (Christopher Iseli)
"""

import pickle, math, time
import numpy as np
import Mapc
from Mapc import mapc
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

## ===== Settings ====##

inputFname = "D:/Honours Data/Dungrove/PointClouds/plot1_ground_removed.mapc"

RAND_STATE = True       # Should a new random state be initialised. Set to False to maintain consistent subsets
N_ITERATIONS = 100      # Number of data splitting iterations
N_TREES = 1000          # Number of trees in RF classifier
TEST_SIZE = 0.3         # Test sample size percentage
N_TOP_FEATURES = 20     # Number of best features to use in classifer

##--------------------##

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

with open(inputFname, "rb") as f:
    pointCloud = pickle.load(f)

segments = pointCloud.df['segment_id'].unique()
num_segments = len(segments)
segment_classes = []


all_attributes = []
spectral_attributes = []
geometric_attributes = []

for segment_id in segments[1:]:
    idxs = pointCloud.df.index[pointCloud.df['segment_id'] == segment_id].tolist()
    class_num = pointCloud.df.at[idxs[0],'class']
    if int(class_num)==1:
        segment_classes.append(1)
    else:
        segment_classes.append(2)

    bright_mean = pointCloud.segments[str(segment_id)]['bright_mean']/65000
    mean_spectra = pointCloud.segments[str(segment_id)]['mean_spectra']/65000
    densities = pointCloud.segments[str(segment_id)]['densities']
    diameters = pointCloud.segments[str(segment_id)]['diameters']
    top_layer_spectra = pointCloud.segments[str(segment_id)]['top_spectra']
    all_attrib_group = [mean_spectra,bright_mean,top_layer_spectra,densities[:],diameters[:]]
    spec_attrib_group = [mean_spectra,bright_mean]
    geo_attrib_group = [densities,diameters]

    all_attributes.append([item for sublist in all_attrib_group for item in sublist])
    spectral_attributes.append([item for sublist in spec_attrib_group for item in sublist])
    geometric_attributes.append([item for sublist in geo_attrib_group for item in sublist])

##===== Get most important features ======##

Y = segment_classes

random_states = np.random.random_integers(0,500,size=N_ITERATIONS)

importances = []

for i in range(N_ITERATIONS):
    X_train, X_test, y_train, y_test = train_test_split(all_attributes, Y, test_size=TEST_SIZE,stratify=Y,random_state=random_states[i])#,random_state=1)

    clf = RandomForestClassifier(oob_score=True,n_estimators=N_TREES,random_state=random_states[i])#,random_state=10)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test,y_test)
    pred_y = clf.predict(X_test)
    importances.append(clf.feature_importances_)

mean_importances = np.mean(importances,axis=0)
sorted_imp_idxs = np.argsort(mean_importances)[::-1]

np.save("influential_idxs.npy",sorted_imp_idxs)

#=== Perform Classification
sorted_imp_idxs =np.load("influential_idxs.npy")
top_features = np.asarray(all_attributes)[:,sorted_imp_idxs[:N_TOP_FEATURES]]

Y=segment_classes
X_sets = [top_features,spectral_attributes,geometric_attributes]

if RAND_STATE:
    random_states = np.random.random_integers(0,500,size=N_ITERATIONS)
    np.save("rand_states.npy",random_states)
else:
    random_states = np.load("rand_states.npy")
class_results = []
cms = []
for n in range(len(X_sets)):
    accuracies = []
    importances = []
    cm = np.zeros((2,2,N_ITERATIONS))
    for i in range(N_ITERATIONS):

        X_train, X_test, y_train, y_test = train_test_split(X_sets[n], Y, test_size=TEST_SIZE,stratify=Y,random_state=random_states[i])

        clf = RandomForestClassifier(oob_score=True,n_estimators=N_TREES,random_state=random_states[i])
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test,y_test)
        pred_y = clf.predict(X_test)
        importances.append(clf.feature_importances_)
        accuracies.append(accuracy)
        cm1=confusion_matrix(y_test,pred_y)
        cm[:,:,i] = cm1
    sum_cm = np.sum(cm,axis=2)
    print(np.min(accuracies),np.mean(accuracies),np.max(accuracies))
    results = list(flatten([np.mean(accuracies,axis=0), np.mean(importances,axis=0)]))
    class_results.append(results)
    cms.append(sum_cm)

np.save("confusion_matrices.npy", np.asarray(cms))
np.save("class_results.npy",np.asarray(class_results))





