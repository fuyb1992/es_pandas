import os
import time
import math
import progressbar
if not progressbar.__version__.startswith('3.'):
    raise Exception('Incorrect version of progerssbar package, please do pip install progressbar2')

import numpy as np
import pandas as pd


from elasticsearch import Elasticsearch
from elasticsearch import helpers


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
    es = Elasticsearch(es_host)
    init_es_tmpl(df, es, doc_type, delete=delete)
    gen = helpers.parallel_bulk(es, (rec_to_actions(df, index, doc_type=doc_type, chunk_size=chunk_size)), thread_count=thread_count,
                        chunk_size=chunk_size, raise_on_error=True, request_timeout=request_timeout)

    success_num = np.sum([res[0] for res in gen])
    rec_num = len(df)
    fail_num = rec_num - success_num

    if (success_num / rec_num) < success_threshold:
        raise Exception('%d records write failed' % fail_num)

    return success_num

def to_pandas(es_host, index, query_rule={'query': {'match_all': {}}}, chunk_size=10000):
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
    es = Elasticsearch(es_host)
    scroll = '5m'
    res = es.search(index=index, body=query_rule, rest_total_hits_as_int=True, size=1, timeout='1m')
    print('Got Hits:', res['hits']['total'])
    if res['hits']['total'] > 0:
        head = res['hits']['hits'][0]['_source'].keys()
    else:
        raise Exception('Empty for %s' % index)

    result = {}
    li = []
    anl = helpers.scan(es, query=query_rule, index=index, scroll=scroll, raise_on_error=True,
                       preserve_order=False, size=chunk_size, clear_scroll=True)

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


def rec_to_actions(df, index, doc_type, chunk_size=1000):
    for i in progressbar.progressbar(range(math.ceil(len(df) / chunk_size))):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(df))
        for record in df.iloc[start_index: end_index, :].to_json(orient='records', date_format='iso', lines=True).split('\n'):
            action = {'_index': index, '_type': doc_type, '_source': record}
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
