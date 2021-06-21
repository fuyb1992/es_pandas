# es_pandas
[![Build Status](https://travis-ci.org/fuyb1992/es_pandas.svg?branch=master)](https://travis-ci.org/fuyb1992/es_pandas) <a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a> [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![PyPi version](https://pypip.in/v/es_pandas/badge.png)](https://crate.io/packages/es_pandas/)
[![Downloads](https://pepy.tech/badge/es-pandas/month)](https://pepy.tech/project/es-pandas/month)

 Read, write and update large scale [pandas](http://pandas.pydata.org/) DataFrame  with [ElasticSearch](https://www.elastic.co/).
 

## Requirements
This package should work on Python3(>=3.4) and ElasticSearch should be version 5.x, 6.x or 7.x.

Installation
The package is hosted on PyPi and can be installed with pip:
```
pip install es_pandas
```
#### Deprecation Notice

Supporting of ElasticSearch 5.x will by deprecated in future version.

## Usage

```
import time

import pandas as pd

from es_pandas import es_pandas


# Information of es cluseter
es_host = 'localhost:9200'
index = 'demo'

# crete es_pandas instance
ep = es_pandas(es_host)

# Example data frame
df = pd.DataFrame({'Num': [x for x in range(100000)]})
df['Alpha'] = 'Hello'
df['Date'] = pd.datetime.now()

# init template if you want
doc_type = 'demo'
ep.init_es_tmpl(df, doc_type)

# Example of write data to es, use the template you create
ep.to_es(df, index, doc_type=doc_type, thread_count=2, chunk_size=10000)

# set use_index=True if you want to use DataFrame index as records' _id
ep.to_es(df, index, doc_type=doc_type, use_index=True, thread_count=2, chunk_size=10000)

# delete records from es
ep.to_es(df.iloc[5000:], index, doc_type=doc_type, _op_type='delete', thread_count=2, chunk_size=10000)

# Update doc by doc _id
df.iloc[:1000, 1] = 'Bye'
df.iloc[:1000, 2] = pd.datetime.now()
ep.to_es(df.iloc[:1000, 1:], index, doc_type=doc_type, _op_type='update')

# Example of read data from es
df = ep.to_pandas(index)
print(df.head())

# return certain fields in es
heads = ['Num', 'Date']
df = ep.to_pandas(index, heads=heads)
print(df.head())

# set certain columns dtype
dtype = {'Num': 'float', 'Alpha': object}
df = ep.to_pandas(index, dtype=dtype)
print(df.dtypes)

# infer dtype from es template
df = ep.to_pandas(index, infer_dtype=True)
print(df.dtypes)

# use query_sql parameter if you want to do query in sql

# Example of write data to es with pandas.io.json
ep.to_es(df, index, doc_type=doc_type, use_pandas_json=True, thread_count=2, chunk_size=10000)
print('write es doc with pandas.io.json finished')
```
