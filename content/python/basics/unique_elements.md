---
title: "Unique_elements"
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
list(df['Values'].unique())
```




    ['A', 'B', 'C']




```python

```
