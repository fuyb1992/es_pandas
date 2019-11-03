# es_pandas
 Read, write and update large scale [pandas](http://pandas.pydata.org/) DataFrame  with [ElasticSearch](https://www.elastic.co/).
 

## Requirements
This package should work on python3(>=3.4) and ElasticSearch should be version 6.x or 7.x(>=6.8).

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
```

## License
(c) 2019 Frank
