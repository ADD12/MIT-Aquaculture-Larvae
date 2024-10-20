# Code used to experiment different classifiers on final training dataset
# Models were simply fit using a subset of the data and then saved to export
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

all_data = pd.read_csv("../../Downloads/official_training_data.csv", index_col=0)
X = all_data[["B","G","R","H", "S", "V"]]
y = all_data["Target"]
y = y-1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
# classifier_1 = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=43)
# classifier_2 = SVC(class_weight = "balanced", random_state=42)
# classifier_3 = LogisticRegression(class_weight='balanced', random_state=14)
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])  
classifier_4 = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=7)
#classifier_5 = BalancedBaggingClassifier(estimator=DecisionTreeClassifier(), random_state=14)
classifier_6 = lgb.LGBMClassifier(is_unbalance=True, random_state=42)
classifier_7 = CatBoostClassifier(auto_class_weights='Balanced', random_state=22, verbose=0)
#classifier_8 = BalancedRandomForestClassifier(n_estimators=100, random_state=4)
#classifier = classifier_8
classifier_4.fit(X_train, y_train)
classifier_6.fit(X_train, y_train)
classifier_7.fit(X_train, y_train)
# y_predictions = classifier_7.predict(X_test)
# cm = confusion_matrix(y_test, y_predictions)
# accuracy = accuracy_score(y_test, y_predictions)
# print(cm)
# print(f"Model Accuracy: {accuracy}")
# classifier_4.save_model('xgboost_model.json')
# classifier_6.booster_.save_model('lightgbm_model.txt')
# classifier_7.save_model('catboost_model.cbm')
