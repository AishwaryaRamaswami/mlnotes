---
title: "Confusion_Matrix"
author: "Aishwarya"
date: 2020-08-10
description: "-"
type: technical_note
draft: false
---

```python
from sklearn.metrics import confusion_matrix
```


```python
C = "Cat"
A = "Ant"
B = "Bird"
```


```python
true = [C, A, C, C, A, B]
pred = [A, A, C, C, A, C]
```


```python
confusion_matrix(true, pred, labels=[A, B, C])
```




    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])




```python

```
