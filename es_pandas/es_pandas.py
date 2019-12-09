import os
import time
import math
import progressbar
from typing import Iterable

if not progressbar.__version__.startswith('3.'):
    raise Exception('Incorrect version of progerssbar package, please do pip install progressbar2')

import numpy as np
import pandas as pd

from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
from elasticsearch.client import IndicesClient, CatClient


class es_pandas(object):
    '''Read, write and update large scale pandas DataFrame with Elasticsearch'''

    def __init__(self, *args, **kwargs):
        self.es = Elasticsearch(*args, **kwargs)
        self.ic = IndicesClient(self.es)
        self.dtype_mapping = {'text': 'category', 'date': 'datetime64[ns]'}
        self.id_col = '_id'
        self.es7 = self.es.info()['version']['number'].startswith('7.')

    def to_es(self, df, index, doc_type=None, use_index=False, thread_count=2, chunk_size=1000, request_timeout=60,
              success_threshold=0.9):
        '''
        :param df: pandas DataFrame data
        :param index: full name of es indices
        :param doc_type: full name of es template
        :param use_index: use DataFrame index as records' _id
        :param delete: delete existing doc_type template if True
        :param thread_count: number of thread sent data to es
        :param chunk_size: number of docs in one chunk sent to es
        :param request_timeout:
        :param success_threshold:
        :return: num of the number of data written into es successfully
        '''
        if self.es7:
            doc_type = '_doc'
        if not doc_type:
            doc_type = index + '_type'
        gen = helpers.parallel_bulk(self.es, (self.rec_to_actions(df, index, doc_type=doc_type, use_index=use_index, chunk_size=chunk_size)),
                                    thread_count=thread_count,
                                    chunk_size=chunk_size, raise_on_error=True, request_timeout=request_timeout)

        success_num = np.sum([res[0] for res in gen])
        rec_num = len(df)
        fail_num = rec_num - success_num

        if (success_num / rec_num) < success_threshold:
            raise Exception('%d records write failed' % fail_num)

        return success_num

    def to_pandas(self, index, query_rule={'query': {'match_all': {}}}, heads=[], dtype={}):
        """
        scroll datas from es, and convert to dataframe, the index of dataframe is from es index,
        about 2 million records/min
        Args:
            es_host: es host ip:port
            query_rule:
            index: full name of es indices
            chunk_size: maximum 10000
            heads: certain columns get from es fields, [] for all fields
            dtype: dict like, pandas dtypes for certain columns
        Returns:
            DataFrame
        """
        scroll = '5m'

        count = self.es.count(index=index, body=query_rule)['count']
        if count < 0:
            raise Exception('Empty for %s' % index)
        if self.es7:
            mapping = self.ic.get_mapping(index=index, include_type_name=False)
        else:
            # Fix es client unrecongnized parameter 'include_type_name' bug for es 6.x
            mapping = self.ic.get_mapping(index=index)
            keys = list(mapping[index]['mappings'].keys())
            if len(keys) > 2: raise Exception('Multi templates exits: %s' % (','.join(keys)))
            tmp = mapping[index]['mappings'][keys[1]]['properties']
            del mapping[index]['mappings'][keys[1]]
            mapping[index]['mappings']['properties'] = tmp
        if len(heads) < 1:
            heads = [k for k in mapping[index]['mappings']['properties'].keys()]
        else:
            unknown_heads = set(heads) - mapping[index]['mappings']['properties'].keys()
            if unknown_heads:
                raise Exception('%s column not found in %s index' % (','.join(unknown_heads), index))

        dtypes = {k: v['type'] for k, v in mapping[index]['mappings']['properties'].items() if k in heads}
        dtypes = {k: self.dtype_mapping[v] for k, v in dtypes.items() if v in self.dtype_mapping}
        if isinstance(dtype, dict):
            dtypes.update(dtype)
        else:
            raise TypeError('dtype_mapping only accept dict')

        query_rule['_source'] = heads
        df_li = defaultdict(list)
        anl = helpers.scan(self.es, query=query_rule, index=index, raise_on_error=True, preserve_order=False,
                           clear_scroll=True)

        for _ in progressbar.progressbar(range(0, count)):
            mes = next(anl)
            for head in heads:
                df_li[head].append(mes['_source'][head])
            df_li[self.id_col].append(mes['_id'])

        return pd.DataFrame(df_li).set_index(self.id_col).astype(dtypes)

    def delete_es(self, df, index, doc_type='_doc', key_col='', chunk_size=1000, thread_count=2):
        '''

        :param df: DataFrame you want to delete from elasticsearch
        :param index: elasticsearch index
        :param doc_type: elasticsearch doc type, _doc default for es 7
        :param key_col: default use DataFrame index
        :param chunk_size:
        :param thread_count:
        :return: number of deleted records
        '''
        assert isinstance(key_col, str)
        if len(key_col):
            df = df.copy().set_index(key_col)
        delete_gen = helpers.parallel_bulk(self.es,
                                           self.rec_to_actions(df, index, doc_type, chunk_size=chunk_size, _op_type='delete'),
                                           thread_count=thread_count, chunk_size=chunk_size)
        return np.sum([res[0] for res in delete_gen])

    def update_es(self, df, index, doc_type, key_col='', chunk_size=1000, thread_count=2):
        '''

        :param df: DataFrame you want to update in elasticsearch
        :param index: elasticsearch index
        :param doc_type: elasticsearch doc type
        :param key_col: default use DataFrame index
        :param chunk_size:
        :param thread_count:
        :return: number of update records
        '''
        assert isinstance(key_col, str)
        raise Exception('to be continued')

    def to_es_dev(self, df, index, key_col, ignore_cols=[], append=False, doc_type=None, delete=False, thread_count=2,
                  chunk_size=1000, request_timeout=60, success_threshold=0.9):
        '''

        :param df: pandas DataFrame data
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

        if not self.ic.exists(index):
            self.to_es(df, index, doc_type=doc_type, delete=delete, thread_count=thread_count,
                       chunk_size=chunk_size, request_timeout=request_timeout, success_threshold=success_threshold)
        else:
            self.update_to_es(df, index, key_col, ignore_cols=ignore_cols, append=append, doc_type=doc_type,
                              thread_count=thread_count, chunk_size=chunk_size, success_threshold=success_threshold)

    def rec_to_actions(self, df, index, doc_type, use_index=False, chunk_size=1000, _op_type=None):
        for i in progressbar.progressbar(range(math.ceil(len(df) / chunk_size))):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, len(df))
            if _op_type == 'update':
                for id, record in zip(df.iloc[start_index: end_index].index.values,
                                      df.iloc[start_index: end_index].to_json(orient='records', date_format='iso',
                                                                                 lines=True).split('\n')):
                    action = {
                        '_op_type': _op_type,
                        '_index': index,
                        '_type': doc_type,
                        '_id': to_int(id),
                        'doc': record
                    }
                    yield action
            elif _op_type == 'delete':
                for id in df.iloc[start_index: end_index, :].index.values:
                    action = {
                        '_op_type': _op_type,
                        '_index': index,
                        '_type': doc_type,
                        '_id': to_int(id)
                    }
                    yield action
            else:
                if use_index:
                    for id, record in zip(df.iloc[start_index: end_index, :].index.values,
                                          df.iloc[start_index: end_index, :].to_json(orient='records', date_format='iso',
                                                                             lines=True).split('\n')):
                        action = {
                            '_index': index,
                            '_type': doc_type,
                            '_id': to_int(id),
                            '_source': record}
                        yield action
                else:
                    for record in df.iloc[start_index: end_index, :].to_json(orient='records', date_format='iso',
                                                                             lines=True).split('\n'):
                        action = {
                            '_index': index,
                            '_type': doc_type,
                            '_source': record}
                        yield action

    def init_es_tmpl(self, df, doc_type, delete=False, shards_count=2, wait_time=5):
        if self.es7:
            return
        tmpl_exits = self.es.indices.exists_template(name=doc_type)
        if tmpl_exits and (not delete):
            return
        columns_body = {}

        if isinstance(df, pd.DataFrame):
            iter_dict = df.dtypes.to_dict()
        elif isinstance(df, dict):
            iter_dict = df
        else:
            raise Exception('init tmpl type is error, only accept DataFrame or dict of head with type mapping')
        for key, data_type in iter_dict.items():
            type_str = getattr(data_type, 'name', data_type)
            if 'int' in type_str:
                columns_body[key] = {'type': 'long'}
            elif 'datetime' in type_str:
                columns_body[key] = {'type': 'date'}
            elif 'float' in type_str:
                columns_body[key] = {'type': 'float'}
            else:
                columns_body[key] = {'type': 'keyword', 'ignore_above': '256'}

        tmpl = {
            'template': '%s*' % doc_type,
            'mappings': {'_default_':
                             {'properties': columns_body}
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
            self.es.indices.delete_template(name=doc_type)
            print('Delete and put template: %s' % doc_type)
        self.es.indices.put_template(name=doc_type, body=tmpl)
        print('New template %s added' % doc_type)
        time.sleep(wait_time)

    def update_to_es(self, df, index, key_col, ignore_cols=[], append=False, doc_type=None, thread_count=2,
                     chunk_size=1000, success_threshold=0.9):
        if self.es7:
            doc_type = '_doc'
        if not doc_type:
            doc_type = index + '_type'
        round = math.ceil(len(df) / chunk_size)
        columns = df.columns.values.tolist()
        change_num = 0
        change_sucess_num = 0
        add_num = 0
        add_sucess_num = 0

        for i in progressbar.progressbar(range(0, round)):
            new_df = df.iloc[i * chunk_size: min((i + 1) * chunk_size, len(df)), :]
            query_rule = {'query': {'terms': {key_col: new_df[key_col].values.tolist()}}}
            old_df = self.to_pandas(index, query_rule=query_rule, heads=columns)
            change_df, _, add_df = self.compare(old_df, new_df, key_col, ignore_cols=ignore_cols)
            change_num += len(change_df)
            add_num += len(add_df)

            if append:
                # to be continued
                pass
            else:
                change_gen = helpers.parallel_bulk(self.es,
                                                   self.rec_to_actions(change_df, index, doc_type,
                                                                       chunk_size=chunk_size, _op_type='delete'),
                                                   thread_count=thread_count, chunk_size=chunk_size)
                add_df = pd.concat([change_df, add_df])
            add_gen = helpers.parallel_bulk(self.es,
                                            self.rec_to_actions(add_df, index, doc_type, chunk_size=chunk_size),
                                            thread_count=thread_count, chunk_size=chunk_size)
            num1, num2 = np.sum([res[0] for res in change_gen]), np.sum([res[0] for res in add_gen])

    def compare(self, old_df, new_df, key_col, ignore_cols=[]):
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

        ignore_cols = set(ignore_cols + [self.id_col, key_col])
        for col in set(old_df.columns.values) - ignore_cols:
            if merge[col].dtype.name == 'category':
                merge[col] = merge[col].astype(object)
            if merge[col].dtype.name == 'datetime64[ns]':
                merge['change'] += abs(merge[col + '_'] - merge[col]) > pd.to_timedelta('1 s')
            else:
                merge['change'] += merge[col + '_'] != merge[col]

        merge = merge.set_index(self.id_col)
        index = merge['change'] > 0
        return merge[index][new_df.columns.values], merge[~index][new_df.columns.values], new_df


def to_int(o):
    if isinstance(o, np.integer): return int(o)
    return o
