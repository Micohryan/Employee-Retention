import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df= pd.read_csv("Employee.csv")
print(df.isnull().sum())
df.info()
df.describe()
df.hist()
plt.show()

corr_matrix = df.corr(numeric_only= True)
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()

for col in df:
  print(df[col].unique())

#split ages into three equal bins
groups = ['Young', 'MiddleAged', 'Adulthood']
df['AgeGroup'] = pd.qcut(df['Age'], q=3, labels=groups)

binary_cols = [
 'Gender',
 'EverBenched']
multi_categories = ['AgeGroup','City','JoiningYear']
ordinal_cat = [['PHD','Masters','Bachelors'],[1,2,3]]

X = df.drop('LeaveOrNot',axis=1)
y= df.LeaveOrNot.values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.preprocessing import LabelEncoder


transformer = ColumnTransformer(transformers=[('ohe1', OneHotEncoder(sparse='False'), multi_categories),
                                             ('oe',OrdinalEncoder(categories=ordinal_cat),['Education','PaymentTier']),
                                             ('ohe2', OneHotEncoder(drop='first',sparse='False'), binary_cols)],remainder='passthrough')

X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)


lrc = LogisticRegression(solver='liblinear', penalty='l1')

knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=7,random_state=2)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=17, random_state=2,max_depth=5)
abc = AdaBoostClassifier(n_estimators=17, random_state=2,learning_rate=0.2)
bc = BaggingClassifier(n_estimators=17, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=18,random_state=2)
xgb = XGBClassifier(n_estimators=17,random_state=2,use_label_encoder=False,eval_metric='mlogloss')


clfs = {
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,zero_division=0)
    
    return accuracy,precision

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
#     print("For ",name)
#     print("Accuracy - ",current_accuracy)
#     print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
print(performance_df)