import os
import time
import math
import progressbar
if not progressbar.__version__.startswith('3.'):
    raise Exception('Incorrect version of progerssbar package, please do pip install progressbar2')

import numpy as np
import pandas as pd


from elasticsearch import Elasticsearch, helpers
from elasticsearch.client import IndicesClient


def to_es(df, es_host, index, doc_type=None, delete=False, thread_count=2, chunk_size=1000, request_timeout=60, success_threshold=0.9):
    '''

    :param df: pandas DataFrame data
    :param es_host: es host ip:port
    :param index: full name of es indices
    :param doc_type: full name of es template
    :param delete: delete existing doc_type template if True
    :param thread_count: number of thread sent data to es
    :param chunk_size: number of docs in one chunk sent to es
    :param request_timeout:
    :param success_threshold:
    :return: num of the number of data written into es successfully
    '''
    if not doc_type:
        doc_type = index + '_type'
    if isinstance(es_host, str):
        es = Elasticsearch(es_host)
    else:
        es = es_host
    if es.info()['version']['number'].startswith('7.'):
        init_es_tmpl(df, es, doc_type, delete=delete)
    gen = helpers.parallel_bulk(es, (rec_to_actions(df, index, doc_type=doc_type, chunk_size=chunk_size)), thread_count=thread_count,
                        chunk_size=chunk_size, raise_on_error=True, request_timeout=request_timeout)

    success_num = np.sum([res[0] for res in gen])
    rec_num = len(df)
    fail_num = rec_num - success_num

    if (success_num / rec_num) < success_threshold:
        raise Exception('%d records write failed' % fail_num)

    return success_num

def to_pandas(es_host, index, query_rule={'query': {'match_all': {}}}, chunk_size=10000, timeout=60):
    """
    scroll datas from es, and convert to dataframe, the index of dataframe is from es index,
    about 2 million records/min
    Args:
        es_host: es host ip:port
        query_rule:
        index: full name of es indices
        chunk_size: maximum 10000
    Returns:
        DataFrame
    """
    if isinstance(es_host, str):
        es = Elasticsearch(es_host)
    else:
        es = es_host
    scroll = '5m'
    res = es.search(index=index, body=query_rule, rest_total_hits_as_int=True, size=1, request_timeout=timeout)
    print('Got Hits:', res['hits']['total'])
    if res['hits']['total'] > 0:
        head = res['hits']['hits'][0]['_source'].keys()
    else:
        raise Exception('Empty for %s' % index)

    result = {}
    li = []
    anl = helpers.scan(es, query=query_rule, index=index, scroll=scroll, raise_on_error=True,
                       preserve_order=False, size=chunk_size, clear_scroll=True, request_timeout=timeout)

    for _ in progressbar.progressbar(range(0, res['hits']['total'] // chunk_size)):
        for _ in range(0, size):
            mes = next(anl)
            result[mes['_id']] = mes['_source'].values()

        li.append(pd.DataFrame.from_dict(result, orient='index', columns=head))
        result = {}

    for _ in progressbar.progressbar(range(0, res['hits']['total'] % chunk_size)):
        mes = next(anl)
        result[mes['_id']] = mes['_source'].values()

    li.append(pd.DataFrame.from_dict(result, orient='index', columns=head))
    return pd.concat(li, axis=0, sort=False)


def to_es_dev(df, es_host, index, key_col, ignore_cols=[], append=False, doc_type=None, delete=False, thread_count=2,
              chunk_size=1000, request_timeout=60, success_threshold=0.9):
    '''

    :param df: pandas DataFrame data
    :param es_host: es host ip:port
    :param index: full name of es indices
    :param doc_type: full name of es template
    :param delete: delete existing doc_type template if True
    :param thread_count: number of thread sent data to es
    :param chunk_size: number of docs in one chunk sent to es
    :param request_timeout:
    :param success_threshold:
    :return: num of the number of data written into es successfully
    '''
    if np.sum(df.duplicated(subset=key_col)) > 0:
        raise Exception('Duplicated data exits in key column: %s' % key_col)

    es = Elasticsearch(es_host, timeout=60)
    ic = IndicesClient(es)

    if not ic.exists(index):
        success_num = to_es(df, es, index, doc_type=doc_type, delete=delete, thread_count=thread_count,
              chunk_size=chunk_size, request_timeout=request_timeout, success_threshold=success_threshold)
    else:
        update_to_es(df, es, index, key_col, ignore_cols=ignore_cols, append=append, doc_type=doc_type,
                     thread_count=thread_count, chunk_size=chunk_size,success_threshold=success_threshold)


def rec_to_actions(df, index, doc_type, chunk_size=1000, update=False, _op_type=None):
    for i in progressbar.progressbar(range(math.ceil(len(df) / chunk_size))):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        if _op_type == 'update':
            for id, record in zip(df.iloc[start_index: end_index, :].index.values,
                                  df.iloc[start_index: end_index, :].to_json(orient='records', date_format='iso', lines=True).split('\n')):
                action = {
                    '_op_type': _op_type,
                    '_index': index,
                    '_type': doc_type,
                    '_id': id,
                    'doc': record
                }
                print(action)
                yield action
        elif _op_type == 'delete':
            for id in df.iloc[start_index: end_index, :].index.values:
                action = {
                    '_op_type': _op_type,
                    '_index': index,
                    '_type': doc_type,
                    '_id': id
                }
                print(action)
                yield action
        else:
            for record in df.iloc[start_index: end_index, :].to_json(orient='records', date_format='iso', lines=True).split('\n'):
                action = {
                    '_index': index,
                    '_type': doc_type,
                    '_source': record}
                yield action


def init_es_tmpl(df, es, doc_type, delete=False, shards_count=2, wait_time=5):
    tmpl_exits = es.indices.exists_template(name=doc_type)
    if tmpl_exits and (not delete):
        return

    columns_body = {}
    for key, data_type in df.dtypes.to_dict().items():
        type_str = data_type.name
        if 'int' in type_str:
            columns_body[key] = {'type': 'long'}
        elif 'datetime' in type_str:
            columns_body[key] = {'type': 'date'}
        elif 'float' in type_str:
            columns_body[key] = {'type': 'float'}
        else:
            columns_body[key] = {'type': 'keyword', 'ignore_above': '256'}

    tmpl = {
        'template': doc_type,
        'mappings': {
            '_default_': {
                'properties': columns_body
            }
        },
        'settings': {
            'index': {
                'refresh_interval': '5s',
                'number_of_shards': shards_count,
                'number_of_replicas': '1',
                'merge': {
                    'scheduler': {
                        'max_thread_count': '1'
                    }
                }
            }
        }
    }

    if tmpl_exits and delete:
        es.indices.delete_template(name=doc_type)
        print('Delete and put template: %s' % doc_type)
    es.indices.put_template(name=doc_type, body=tmpl)
    print('New template %s added' % doc_type)
    time.sleep(wait_time)


def update_to_es(df, es_host, index, key_col, ignore_cols=[], append=False, doc_type=None, thread_count=2,
              chunk_size=1000, success_threshold=0.9):
    if not doc_type:
        doc_type = index + '_type'
    if isinstance(es_host, str):
        es = Elasticsearch(es_host)
    else:
        es = es_host
    round = math.ceil(len(df) / chunk_size)
    columns = df.columns.values.tolist()
    time_colunms = [col for col in columns if df[col].dtype.name == 'datetime64[ns]']

    change_num = 0
    change_sucess_num = 0
    add_num = 0
    add_sucess_num = 0

    for i in progressbar.progressbar(range(0, round)):
        new_df = df.iloc[i * chunk_size: min((i + 1) * chunk_size, len(df)), :]
        query_rule = {'query': {'terms': {key_col: new_df[key_col].values.tolist()}}, '_source': columns}
        old_df = to_pandas(es, index, query_rule=query_rule, chunk_size=chunk_size)
        for col in time_colunms:
            old_df[col] = old_df[col].astype('datetime64[ns]')

        change_df, _ , add_df = compare(old_df, new_df, key_col, ignore_cols=ignore_cols)
        change_num += len(change_df)
        add_num += len(add_df)

        if append:
            # to be continued
            pass
        else:
            change_gen = helpers.parallel_bulk(es, rec_to_actions(change_df, index, doc_type, chunk_size=chunk_size, _op_type='delete'),
                                               thread_count=thread_count, chunk_size=chunk_size)
            add_df = pd.concat([change_df, add_df])
        add_gen = helpers.parallel_bulk(es, rec_to_actions(add_df, index, doc_type, chunk_size=chunk_size),
                                        thread_count=thread_count, chunk_size=chunk_size)
        num1, num2 = np.sum([res[0] for res in change_gen]), np.sum([res[0] for res in add_gen])


def compare(old_df, new_df, key_col, ignore_cols=[]):
    """

    :param old_df: old DataFrame
    :param new_df: new DataFrame
    :param key: unique key column name, str
    :param ingore_cols: compare ignore columns names, list
    :return:
    """
    assert isinstance(key_col, str)
    assert isinstance(ignore_cols, list)

    merge = pd.merge(old_df.reset_index(), new_df, on=key_col, how='inner', left_index=True, suffixes=('_', ''))
    # drop merged rows
    new_df = new_df.copy()
    new_df.drop(merge.index, inplace=True)
    merge['change'] = 0

    ignore_cols = set(ignore_cols + ['index'] + [key_col])
    print(ignore_cols)
    for col in set(old_df.columns.values) - ignore_cols:
        if merge[col].dtype.name == 'category':
            merge[col] = merge[col].astype(object)
        if merge[col].dtype.name == 'datetime64[ns]':
            merge['change'] += abs(merge[col + '_'] - merge[col]) > pd.to_timedelta('1 s')
        else:
            merge[[col, col + '_']] = merge[[col, col + '_']].fillna(0)
            merge['change'] += merge[col + '_'] != merge[col]

    merge = merge.set_index('index')
    index = merge['change'] > 0
    return merge[index][new_df.columns.values], merge[~index][new_df.columns.values], new_df