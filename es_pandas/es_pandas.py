import re
import warnings
import progressbar

if not progressbar.__version__.startswith('3.'):
    raise Exception('Incorrect version of progerssbar package, please do pip install progressbar2')

import numpy as np
import pandas as pd

from pandas.io import json
from elasticsearch import Elasticsearch, helpers


class es_pandas(object):
    '''Read, write and update large scale pandas DataFrame with Elasticsearch'''

    def __init__(self, *args, **kwargs):
        self.es = Elasticsearch(*args, **kwargs)
        self.ic = self.es.indices
        self.dtype_mapping = {'text': 'category', 'date': 'datetime64[ns]'}
        self.id_col = '_id'
        self.es_version_str = self.es.info()['version']['number']
        self.es_version = [int(x) for x in re.findall("[0-9]+", self.es_version_str)]
        if self.es_version[0] < 6:
            warnings.warn('Supporting of ElasticSearch 5.x will by deprecated in future version, '
                          'current es version: %s' % self.es_version_str, category=FutureWarning)

    def to_es(self, df, index, doc_type=None, use_index=False, show_progress=True, 
              success_threshold=0.9, _op_type='index', use_pandas_json=False, date_format='iso', **kwargs):
        '''
        :param df: pandas DataFrame data
        :param index: full name of es indices
        :param doc_type: full name of es template
        :param use_index: use DataFrame index as records' _id
        :param success_threshold:
        :param _op_type: elasticsearch _op_type, default 'index', choices: 'index', 'create', 'update', 'delete'
        :param use_pandas_json: default False, if True, use pandas.io.json serialize
        :param date_format: default iso, only works when use_pandas_json=True
        :return: num of the number of data written into es successfully
        '''
        if self.es_version[0] > 6:
            doc_type = None
        elif self.es_version[0] > 5:
            doc_type = '_doc'
        elif not doc_type:
            doc_type = index + '_type'
        gen = helpers.parallel_bulk(self.es,
                                    (self.rec_to_actions(df, index, doc_type=doc_type, show_progress=show_progress, 
                                                         use_index=use_index, _op_type=_op_type,
                                                         use_pandas_json=use_pandas_json, date_format=date_format)),
                                    **kwargs)

        success_num = np.sum([res[0] for res in gen])
        rec_num = len(df)
        fail_num = rec_num - success_num

        if (success_num / rec_num) < success_threshold:
            raise Exception('%d records write failed' % fail_num)

        return success_num

    def get_source(self, anl, show_progress=False, count=0):
        if show_progress:
            with progressbar.ProgressBar(max_value=count) as bar:
                for i in range(count):
                    mes = next(anl)
                    yield {'_id': mes['_id'], **mes['_source']}
                    bar.update(i)
        else:
            for mes in anl:
                yield {'_id': mes['_id'], **mes['_source']}

    def infer_dtype(self, index, heads):
        if self.es_version[0] > 6:
            mapping = self.ic.get_mapping(index=index)
        else:
            # Fix es client unrecongnized parameter 'include_type_name' bug for es 6.x
            mapping = self.ic.get_mapping(index=index)
            key = [k for k in mapping[index]['mappings'].keys() if k != '_default_']
            if len(key) < 1: raise Exception('No templates exits: %s' % index)
            mapping[index]['mappings']['properties'] = mapping[index]['mappings'][key[0]]['properties']
        dtype = {k: v['type'] for k, v in mapping[index]['mappings']['properties'].items() if k in heads}
        dtype = {k: self.dtype_mapping[v] for k, v in dtype.items() if v in self.dtype_mapping}
        return dtype


    def to_pandas(self, index, query_rule=None, heads=[], dtype={}, infer_dtype=False, show_progress=True, query_sql=None, **kwargs):
        """
        scroll datas from es, and convert to dataframe, the index of dataframe is from es index,
        about 2 million records/min
        Args:
            index: full name of es indices
            query_rule: dict, default match_all, elasticsearch query DSL
            heads: certain columns get from es fields, [] for all fields
            dtype: dict like, pandas dtypes for certain columns
            infer_dtype: bool, default False, if true, get dtype from es template
            show_progress: bool, default True, if true, show progressbar on console
            query_sql: string or dict, default None, SQL containing query to filter
        Returns:
            DataFrame
        """
        if query_sql:
            if isinstance(query_sql, str):
                dsl_from_sql = self.es.sql.translate({'query': query_sql})
            elif isinstance(query_sql, dict):
                dsl_from_sql = self.es.sql.translate(query_sql)
            else:
                raise Exception('Parameter data type error, query_sql should be string or dict type')
            if query_rule:
                raise Exception('Cannot use query_rule and query_sql at the same time')
            else:
                query_rule = {'query': dsl_from_sql['query']}
        elif not query_rule:
            query_rule = {'query': {'match_all': {}}}
        count = self.es.count(index=index, body=query_rule)['count']
        if count < 1:
            return pd.DataFrame()
        query_rule['_source'] = heads
        anl = helpers.scan(self.es, query=query_rule, index=index, **kwargs)
        df = pd.DataFrame(self.get_source(anl, show_progress=show_progress, count=count)).set_index('_id')
        if infer_dtype:
            dtype = self.infer_dtype(index, df.columns.values)
        if len(dtype):
            df = df.astype(dtype)
        return df

    @staticmethod
    def serialize(row, columns, use_pandas_json, iso_dates):
        if use_pandas_json:
            return json.dumps(dict(zip(columns, row)), iso_dates=iso_dates)
        return dict(zip(columns, [None if np.all(pd.isna(r)) else r for r in row]))

    @staticmethod
    def gen_action(**kwargs):
        return {k: v for k, v in kwargs.items() if v is not None}

    def rec_to_actions(self, df, index, doc_type=None, use_index=False, _op_type='index', use_pandas_json=False, date_format='iso', show_progress=True):
        if show_progress:
            bar = progressbar.ProgressBar(max_value=df.shape[0])
        else:
            bar = BarNothing()
        columns = df.columns.tolist()
        iso_dates = date_format == 'iso'
        if use_index and (_op_type in ['create', 'index']):
            for i, row in enumerate(df.itertuples(name=None, index=use_index)):
                bar.update(i)
                _id = row[0]
                record = self.serialize(row[1:], columns, use_pandas_json, iso_dates)
                action = self.gen_action(_op_type=_op_type, _index=index, _type=doc_type, _id=_id, _source=record)
                yield action
        elif (not use_index) and (_op_type == 'index'):
            for i, row in enumerate(df.itertuples(name=None, index=use_index)):
                bar.update(i)
                record = self.serialize(row, columns, use_pandas_json, iso_dates)
                action = self.gen_action(_op_type=_op_type, _index=index, _type=doc_type, _source=record)
                yield action
        elif _op_type == 'update':
            for i, row in enumerate(df.itertuples(name=None, index=True)):
                bar.update(i)
                _id = row[0]
                record = self.serialize(row[1:], columns, False, iso_dates)
                action = self.gen_action(_op_type=_op_type, _index=index, _type=doc_type, _id=_id, doc=record)
                yield action
        elif _op_type == 'delete':
            for i, _id in enumerate(df.index.values.tolist()):
                bar.update(i)
                action = self.gen_action(_op_type=_op_type, _index=index, _type=doc_type, _id=_id)
                yield action
        else:
            raise Exception('[%s] action with %s using index not supported' % (_op_type, '' if use_index else 'not'))

    def init_es_tmpl(self, df, doc_type, delete=False, index_patterns=None, **kwargs):
        '''

        :param df: pd.DataFrame
        :param doc_type: str, name of doc_type
        :param delete: bool, if True, delete existed template
        :param index_patterns: list, default None, [doc_type*]
        :param kwargs: kwargs for template settings,
               example: number_of_shards, number_of_replicas, refresh_interval
        :return:
        '''
        tmpl_exits = self.es.indices.exists_template(name=doc_type)
        if tmpl_exits and (not delete):
            return
        if index_patterns is None:
            index_patterns = ['%s*' % doc_type]
        columns_body = {}

        if isinstance(df, pd.DataFrame):
            iter_dict = df.dtypes.to_dict()
        elif isinstance(df, dict):
            iter_dict = df
        else:
            raise Exception('init tmpl type is error, only accept DataFrame or dict of head with type mapping')
        for key, data_type in iter_dict.items():
            type_str = getattr(data_type, 'name', data_type).lower()
            if 'int' in type_str:
                columns_body[key] = {'type': 'long'}
            elif 'datetime' in type_str:
                columns_body[key] = {'type': 'date'}
            elif 'float' in type_str:
                columns_body[key] = {'type': 'float'}
            else:
                columns_body[key] = {'type': 'keyword', 'ignore_above': '256'}

        tmpl = {
            'index_patterns': index_patterns,
            'settings': {**kwargs}
        }
        if self.es_version[0] > 6:
            tmpl['mappings'] = {'properties': columns_body}
        elif self.es_version[0] > 5:
            tmpl['mappings'] = {'_doc': {'properties': columns_body}}
        else:
            tmpl['mappings'] = {'_default_': {'properties': columns_body}}
        if tmpl_exits and delete:
            self.es.indices.delete_template(name=doc_type)
            print('Delete and put template: %s' % doc_type)
        self.es.indices.put_template(name=doc_type, body=tmpl)
        print('New template %s added' % doc_type)


class BarNothing(object):
    def update(self, arg):
        pass
