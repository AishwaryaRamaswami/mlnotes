---
title: "Remove_stopwords"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
import pandas as pd
import nltk
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/aishwarya/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True




```python
Sms_content=['hi,how are you','I am fine','myself aishwarya?']
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
      <td>I am fine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>myself aishwarya?</td>
    </tr>
  </tbody>
</table>
</div>




```python
stopwords=nltk.corpus.stopwords.words('english')
stopwords[:5]
```




    ['i', 'me', 'my', 'myself', 'we']




```python
def remove_stopwords(text):
    clean_text=[word for word in text if word not in stopwords]
    return clean_text
```


```python
df['clean_text'] = df['sms'].apply(lambda row : remove_stopwords(row))
df.head()
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
      <th>clean_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hi,how are you</td>
      <td>[h, ,, h, w,  , r, e,  , u]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I am fine</td>
      <td>[I,  ,  , f, n, e]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>myself aishwarya?</td>
      <td>[e, l, f,  , h, w, r, ?]</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
