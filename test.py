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
# add null value
df.iloc[0] = None

# init template if you want
doc_type = 'demo'
ep.init_es_tmpl(df, doc_type, delete=True)

# Example of write data to es
ep.to_es(df, index, doc_type=doc_type, thread_count=2, chunk_size=10000)
print('write es doc without index finished')

# Example of use DataFrame index as es doc _id
ep.to_es(df, index, doc_type=doc_type, use_index=True, thread_count=2, chunk_size=10000)
print('write es doc with index finished')

# waiting for es data writing
time.sleep(5)

# Delete doc by doc _id
ep.to_es(df.iloc[5000:], index, doc_type=doc_type, _op_type='delete', thread_count=2, chunk_size=10000)
print('delete es doc finished')

# waiting for es data writing
time.sleep(5)

# Update doc by doc _id
df.iloc[:1000, 1] = 'Bye'
df.iloc[:1000, 2] = pd.datetime.now()
ep.to_es(df.iloc[:1000, 1:], index, doc_type=doc_type, _op_type='update', thread_count=2, chunk_size=10000)
print('update es doc finished')

# waiting for es data writing
time.sleep(5)

# get certain fields from es, set certain columns dtype
heads = ['Num', 'Date', 'Alpha']
dtype = {'Num': 'float', 'Alpha': object}
df = ep.to_pandas(index, heads=heads, dtype=dtype)
print(df.head())
print(df.dtypes)

# infer dtypes from es template
df = ep.to_pandas(index, infer_dtype=True)
print(df.dtypes)

# Example of write data to es with pandas.io.json
ep.to_es(df, index, doc_type=doc_type, use_pandas_json=True, thread_count=2, chunk_size=10000)
print('write es doc with pandas.io.json finished')
