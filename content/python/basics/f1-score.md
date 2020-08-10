---
title: "F1-Score"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
from sklearn.metrics import f1_score
```


```python
true = [0, 1, 2, 0, 1, 2]
pred = [0, 2, 1, 0, 0, 1]
```


```python
f1_score(true, pred,average=None).mean()
```




    0.26666666666666666




```python
f1_score(true, pred,average="macro")
```




    0.26666666666666666




```python
f1_score(true, pred,average="micro")
```




    0.3333333333333333




```python
f1_score(true, pred,average="weighted")
```




    0.26666666666666666




```python

```
