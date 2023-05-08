import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


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
def count_plot(data, x:str , hue:str, Title:str, labels:list = None) -> None:
    sns.countplot(data = data, x = x , hue = hue).set(title=Title)
    plt.legend(labels = labels)
    plt.show()
    return

count_plot(df, x = "JoiningYear", hue = "LeaveOrNot",Title = 'Joinning Year vs LeaveorNot', labels=['Left', 'Stayed'])
count_plot(df, x = "Education", hue = "LeaveOrNot", Title = 'Education vs LeaveorNot', labels=['Left', 'Stayed'])
count_plot(df, x = "EverBenched", hue = "LeaveOrNot",Title = 'EverBenched vs LeaveorNot', labels=['Left', 'Stayed'])
count_plot(df, x = "Gender", hue = "LeaveOrNot",Title = 'Gender vs LeaveorNot', labels=['Left', 'Stayed'])
count_plot(df, x = "Gender", hue = "Education",Title = 'Gender vs Education')
count_plot(df, x = "Age", hue = "LeaveOrNot",Title = 'Age vs LeaveorNot', labels=['Left', 'Stayed'])

