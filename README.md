# es_pandas
<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a> [![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE) [![PyPi version](https://pypip.in/v/es_pandas/badge.png)](https://crate.io/packages/es_pandas/)
[![PyPi downloads](https://pypip.in/d/es_pandas/badge.png)](https://crate.io/packages/$es_pandas/)

 Read, write and update large scale [pandas](http://pandas.pydata.org/) DataFrame  with [ElasticSearch](https://www.elastic.co/).
 

## Requirements
This package should work on Python3(>=3.4) and ElasticSearch should be version 6.x or 7.x(>=6.8).

Installation
The package is hosted on PyPi and can be installed with pip:
```
pip install es_pandas
```
## Usage

```
import time

import pandas as pd

from es_pandas import to_pandas, to_es


# Information of es cluseter
es_host = 'localhost:9200'
index = 'demo'

# Example data frame
df = pd.DataFrame({'Alpha': [chr(i) for i in range(97, 128)], 
                    'Num': [x for x in range(31)], 
                    'Date': pd.date_range(start='2019/01/01', end='2019/01/31')})

# Example of write data to es, auto create and put template to es if template does not exits
to_es(df, es_host, index)

time.sleep(10)
# Example of read data from es
df = to_pandas(es_host, index)
print(df.head())
# only return particular fields in es
heads = ['Num', 'Date']
df = to_pandas(es_host, index, heads=heads)
print(df.head()

df2 = pd.DataFrame({'Alpha': [chr(i) for i in range(97, 129)],
                    'Num': [x for x in range(32)],
                    'Date': pd.date_range(start='2019/01/01', end='2019/02/01')})

df2.loc[df2['Num']==10, ['Alpha']] = 'change'

# Example of update data in es
to_es_dev(df2, es_host, index, 'Num')
```
### More about update
`to_es_dev(df, es_host, index, key_col, ignore_cols=[])` function is available if you want to write or update data with ElasticSearch.

`to_es_dev` function writes `df` to es if `index` not exits, or it reads data from ElasticSearch in batches, and compare the data with `df` by merging them on `key_col`, if you want to ignore some columns when comparing, set it with `ignore_col` parameter. Moreover, new records in `df` will be written to ElasticSearch.

## License
(c) 2019 Frank
