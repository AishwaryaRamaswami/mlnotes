---
title: "Encoding"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
import pandas as pd
```


```python
list1 = ['A','B','B','A','C']
df=pd.DataFrame(list1,columns={'Values'})
df
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
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn import preprocessing
encode = preprocessing.LabelEncoder()
df['Encoded_Value'] = encode.fit_transform(df['Values'])
df
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
      <th>Values</th>
      <th>Encoded_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
