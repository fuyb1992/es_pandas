import time

import pandas as pd

from es_pandas import es_pandas


# Information of es cluseter
es_host = 'localhost:9200'
index = 'demo'

# crete es_pandas instance
ep = es_pandas(es_host)

# Example data frame
df = pd.DataFrame({'Alpha': [chr(i) for i in range(97, 128)],
                    'Num': [x for x in range(31)],
                    'Date': pd.date_range(start='2019/01/01', end='2019/01/31')})

# init template if you want
doc_type = 'demo'
ep.init_es_tmpl(df, doc_type)
# Example of write data to es, auto create and put template to es if template does not exits
ep.to_es(df, index)

# waiting for es data writing
time.sleep(10)

# return certain fields in es
heads = ['Num', 'Date']
df = ep.to_pandas(index, heads=heads)

# set certain columns dtype
dtype = {'Num': 'float', 'Alpha': object}
df = ep.to_pandas(index, dtype=dtype)

# Example of read data from es
df = ep.to_pandas(index)

df2 = pd.DataFrame({'Alpha': [chr(i) for i in range(97, 129)],
                    'Num': [x for x in range(32)],
                    'Date': pd.date_range(start='2019/01/01', end='2019/02/01')})

df2.loc[df2['Num']==10, ['Alpha']] = 'change'

# Example of update data in es
ep.to_es_dev(df2, index, 'Num')
