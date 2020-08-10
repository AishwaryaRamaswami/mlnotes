---
title: "Remove_Punctuation"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
import string
import pandas as pd
```


```python
Sms_content=['hi,how are you','Iam fine','what is it?']
df=pd.DataFrame(Sms_content,columns={'sms'})
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
      <th>sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi,how are you</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iam fine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what is it?</td>
    </tr>
  </tbody>
</table>
</div>




```python
def remove_punctuation(text):
    new_text=''.join([char for char in text if char not in string.punctuation])
    return new_text

```


```python
df['new_sms']=df['sms'].apply(lambda row : remove_punctuation(row))
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
      <th>sms</th>
      <th>new_sms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi,how are you</td>
      <td>hihow are you</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Iam fine</td>
      <td>Iam fine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>what is it?</td>
      <td>what is it</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
