---
title: "Stemmer"
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
ps = nltk.PorterStemmer()
```


```python
e_words= ["wait", "waiting", "waited", "waits"]
for w in e_words:
    rootWord=ps.stem(w)
    print(rootWord)
```

    wait
    wait
    wait
    wait



```python

```
