

```python
1+1
```




    2




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

```


```python
SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150
```


```python
sdss_df = pd.read_csv('C:/Users/reaction/Desktop/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
```


```python
sdss_df.head()
#sdss_df.info()
```





</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objid</th>
      <th>ra</th>
      <th>dec</th>
      <th>u</th>
      <th>g</th>
      <th>r</th>
      <th>i</th>
      <th>z</th>
      <th>run</th>
      <th>rerun</th>
      <th>camcol</th>
      <th>field</th>
      <th>specobjid</th>
      <th>class</th>
      <th>redshift</th>
      <th>plate</th>
      <th>mjd</th>
      <th>fiberid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.237650e+18</td>
      <td>183.531326</td>
      <td>0.089693</td>
      <td>19.47406</td>
      <td>17.04240</td>
      <td>15.94699</td>
      <td>15.50342</td>
      <td>15.22531</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>267</td>
      <td>3.722360e+18</td>
      <td>STAR</td>
      <td>-0.000009</td>
      <td>3306</td>
      <td>54922</td>
      <td>491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.237650e+18</td>
      <td>183.598371</td>
      <td>0.135285</td>
      <td>18.66280</td>
      <td>17.21449</td>
      <td>16.67637</td>
      <td>16.48922</td>
      <td>16.39150</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>267</td>
      <td>3.638140e+17</td>
      <td>STAR</td>
      <td>-0.000055</td>
      <td>323</td>
      <td>51615</td>
      <td>541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.237650e+18</td>
      <td>183.680207</td>
      <td>0.126185</td>
      <td>19.38298</td>
      <td>18.19169</td>
      <td>17.47428</td>
      <td>17.08732</td>
      <td>16.80125</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>268</td>
      <td>3.232740e+17</td>
      <td>GALAXY</td>
      <td>0.123111</td>
      <td>287</td>
      <td>52023</td>
      <td>513</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.237650e+18</td>
      <td>183.870529</td>
      <td>0.049911</td>
      <td>17.76536</td>
      <td>16.60272</td>
      <td>16.16116</td>
      <td>15.98233</td>
      <td>15.90438</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>269</td>
      <td>3.722370e+18</td>
      <td>STAR</td>
      <td>-0.000111</td>
      <td>3306</td>
      <td>54922</td>
      <td>510</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.237650e+18</td>
      <td>183.883288</td>
      <td>0.102557</td>
      <td>17.55025</td>
      <td>16.26342</td>
      <td>16.43869</td>
      <td>16.55492</td>
      <td>16.61326</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>269</td>
      <td>3.722370e+18</td>
      <td>STAR</td>
      <td>0.000590</td>
      <td>3306</td>
      <td>54922</td>
      <td>512</td>
    </tr>
  </tbody>
</table>
</div>




```python
sdss_df['class'].value_counts()
```




    GALAXY    4998
    STAR      4152
    QSO        850
    Name: class, dtype: int64




```python
sdss_df.columns.values
```




    array(['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun',
           'camcol', 'field', 'specobjid', 'class', 'redshift', 'plate',
           'mjd', 'fiberid'], dtype=object)




```python
sdss_df.columns
```




    Index(['objid', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'run', 'rerun', 'camcol',
           'field', 'specobjid', 'class', 'redshift', 'plate', 'mjd', 'fiberid'],
          dtype='object')




```python
sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)
sdss_df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ra</th>
      <th>dec</th>
      <th>u</th>
      <th>g</th>
      <th>r</th>
      <th>i</th>
      <th>z</th>
      <th>class</th>
      <th>redshift</th>
      <th>plate</th>
      <th>mjd</th>
      <th>fiberid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>183.531326</td>
      <td>0.089693</td>
      <td>19.47406</td>
      <td>17.0424</td>
      <td>15.94699</td>
      <td>15.50342</td>
      <td>15.22531</td>
      <td>STAR</td>
      <td>-0.000009</td>
      <td>3306</td>
      <td>54922</td>
      <td>491</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(sdss_df[sdss_df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)
ax.set_title('Star')
ax = sns.distplot(sdss_df[sdss_df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)
ax.set_title('Galaxy')
ax = sns.distplot(sdss_df[sdss_df['class']=='QSO'].redshift, bins = 30, ax = axes[2], kde = False)
ax = ax.set_title('QSO')
```

    C:\Users\reaction\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    C:\Users\reaction\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    C:\Users\reaction\Anaconda3\lib\site-packages\matplotlib\axes\_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "
    


![png](output_9_1.png)



```python

```




    Text(0.5,1,'Equatorial coordinates')




![png](output_10_1.png)



```python
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
sdss_df_fe = sdss_df
le = LabelEncoder()
y_encoded = le.fit_transform(sdss_df_fe['class'])
sdss_df_fe['class'] = y_encoded
X_train, X_test, y_train, y_test = train_test_split(sdss_df_fe.drop('class', axis=1), 
                                                    sdss_df_fe['class'], test_size=0.33)
import time
```


```python
xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))

```

    XGBoost's prediction accuracy is: 99.06
    Time consumed for training: 1.803
    Time consumed for prediction: 0.02896 seconds
    

    C:\Users\reaction\Anaconda3\lib\site-packages\sklearn\preprocessing\label.py:151: DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty.
      if diff:
    
