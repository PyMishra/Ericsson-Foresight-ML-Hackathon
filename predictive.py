#%%
import pandas as pd
import re
import numpy as np
from math import floor
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
#%%
df = pd.read_csv('train_file.csv')
df.head()
#%%
df_test = pd.read_csv('test_file.csv')

#%%
df.info()

#%%
df['Creator'].fillna('Unknown', inplace=True)
df['Subjects'].fillna('Unknown', inplace=True)
df['Publisher'].fillna('Unknown', inplace=True)
df.info()
#%%
df_test['Creator'].fillna('Unknown', inplace=True)
df_test['Subjects'].fillna('Unknown', inplace=True)
df_test['Publisher'].fillna('Unknown', inplace=True)

#%%
df['PublicationYear'].replace('[date of publication not identified]', '0', inplace=True)
df_test['PublicationYear'].replace('[date of publication not identified]', '0', inplace=True)
#%%
pubyr = list(df['PublicationYear'].fillna('0'))
pubyr_test = list(df_test['PublicationYear'].fillna('0'))
#%%
type(pubyr[0])
search.group()+'0'
#%%
pubyr_fixed = []
for yr in pubyr:
    if yr == '0':
        pubyr_fixed.append('0')
        continue
    search = re.search(r'\d{4}', yr)
    if search:
        pubyr_fixed.append(int(search.group()))
        continue
    search = re.search(r'\d{3}', yr)
    if search:
        pubyr_fixed.append(int(search.group() + '0'))
        continue
    search = re.search(r'\d{2}', yr)
    if search:
        pubyr_fixed.append(int(search.group() + '00'))
    
#%%
df.info()
len(pubyr_fixed)
#%%
df['PublicationYear'] = pubyr_fixed
df.info()

#%%
pubyr_fixed = []
for yr in pubyr_test:
    if yr == '0':
        pubyr_fixed.append('0')
        continue
    search = re.search(r'\d{4}', yr)
    if search:
        pubyr_fixed.append(int(search.group()))
        continue
    search = re.search(r'\d{3}', yr)
    if search:
        pubyr_fixed.append(int(search.group() + '0'))
        continue
    search = re.search(r'\d{2}', yr)
    if search:
        pubyr_fixed.append(int(search.group() + '00'))

#%%
df_test['PublicationYear'] = pubyr_fixed
df_test.info()

#%%
df.head(50)

#%%
df.drop(['ID', 'UsageClass', 'CheckoutType', 'CheckoutYear', 'CheckoutMonth'], axis=1, inplace=True)
df_test.drop(['ID', 'UsageClass', 'CheckoutType', 'CheckoutYear', 'CheckoutMonth'], axis=1, inplace=True)
df.info()

#%%
def book_func(data):
    keywords = ['literature', 'book']
    for key in keywords:
        if key in data:
            return True
    return False


#%%
# Filling type with book if the word book found in title
df['Type_Book'] = df['Title'].apply(lambda data : book_func(data))
df_test['Type_Book'] = df_test['Title'].apply(lambda data : book_func(data))
print(df['Type_Book'].value_counts())
print(df_test['Type_Book'].value_counts())

#%%
x_train = df.drop(['Title', 'Subjects', 'MaterialType'], axis=1)
y_train = df['MaterialType']
x_train.head()

#%%
y_train.head()

#%%
x_test = df_test.drop(['Title', 'Subjects'], axis=1)
x_test.head()

#%%
x_train['PublicationYear'].replace('0', np.nan, inplace=True)
x_train['PublicationYear'].fillna(x_train['PublicationYear'].mean(), inplace=True)
#%%
x_test['PublicationYear'].replace('0', np.nan, inplace=True)
x_test['PublicationYear'].fillna(x_train['PublicationYear'].mean(), inplace=True)

#%%
x_train.drop(['Creator', 'Publisher'], axis=1, inplace=True)
x_test.drop(['Creator', 'Publisher'], axis=1, inplace=True)
#%%
model = DecisionTreeClassifier(max_depth=5)
model.fit(x_train, y_train)
#%%
y_pred = model.predict(x_test)

#%%
y_pred

#%%
prediction = pd.DataFrame({'ID': df_test['ID'], 'MaterialType':y_pred})
prediction.to_csv('Submission.csv', index=False)

#%%
