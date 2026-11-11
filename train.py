import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
p=argparse.ArgumentParser(); p.add_argument('--data',default='IRIS.csv'); args=p.parse_args()
df=pd.read_csv(args.data)
features=['sepal_length','sepal_width','petal_length','petal_width']
X_train,X_test,y_train,y_test=train_test_split(df[features],df.species,test_size=.2,random_state=42,stratify=df.species)
model=Pipeline([('impute',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
model.fit(X_train,y_train)
baseline=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
print(json.dumps({'test_rows':len(y_test),'accuracy':accuracy_score(y_test,model.predict(X_test)),'baseline_accuracy':accuracy_score(y_test,baseline.predict(X_test))},indent=2))
print(classification_report(y_test,model.predict(X_test),zero_division=0))
