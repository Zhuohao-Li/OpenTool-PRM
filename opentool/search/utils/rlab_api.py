import json
import logging
import os
import pickle
import re
import requests
import traceback
# import parse_json

RLAB_API_HEADERS = {
    'Content-Type': 'application/json',
    'rlab-api-key': 'ICBU_H1425_95682FCB-8E8D-4F2F-BFDD-86CD0FE87DD4C71',
    'rlab-request-source': 'sourcing-agent',
}

RLAB_QP_ID = 'ICBU_N4552_158606E84'  # 之后只有这一个QP服务
RLAB_KNOWLEDGE_ID = 'ICBU_N4552_220211C13'  # 类目知识，新Rlab，预发测试中
# RLAB_KNOWLEDGE_ID = 'ICBU_L5736_1528256C4'  # 类目知识，老Rlab

RLAB_MANUFACTURE_NAME_RECOGNIZE = 'ICBU_P0715_264433F4D'

RLAB_CATE_CVR_INFO = 'ICBU_P0715_2604351AF'  # 宽泛类目下类目转化信息
RLAB_USER_QUERIES_ID = 'ICBU_P0715_2612325FD'  # 用户近期历史会话

RLAB_CATEINFO_ID = 'ICBU_L5736_182847C0A'  # 类目信息
RLAB_QWEN15_72B_ID = 'ICBU_L5736_90667059'  # Qwen-1.5-72B
RLAB_MSG_SAVE_ID = 'ICBU_N4552_156603F2F'
RLAB_MSG_FETCH_ID = 'ICBU_N4552_154608325'

RLAB_NVGT_CCH_ID = 'ICBU_P0715_240825BB1'
RLAB_ONLINE_CACHE_ID = 'ICBU_N3984_254002E2F'  # 在线读写缓存

RLAB_ISPRTN_FOR_NVGT = 'ICBU_P0715_253013025'  # 场景搜接入Inspiration数据

RLAB_PROD_PV_READ = 'ICBU_L5736_17720228F'  # 获取商品PV
RLAB_PROD_INFO = 'ICBU_L5736_2338040CB'  # 获取商品信息（Agent）
RLAB_PROD_INFO_BASIC = 'ICBU_L5736_140234B77'  # 获取商品信息（基础）
RLAB_PROD_DETAIL = 'ICBU_P0715_241402A6F'  # 获取商详页（卖家上传）

RLAB_PICTURE_MATCH = 'ICBU_P0715_241212023'  # 获取图文匹配详情
RLAB_PICTURE_QUERYGEN = 'ICBU_P0715_24202395D'  # 获取图文生成query
RLAB_DETAIL_QUERYGEN = 'ICBU_P0715_24202270B'  # 获取商详页生成query
RLAB_COMPARE_QUERYGEN = 'ICBU_P0715_2430060C0'  # 获取商详页生成query
RLAB_LOCATION_DATA = 'ICBU_N3984_2414136C0'  # 根据location ID获取Location的内容
RLAB_SUMMARY_GEN = 'ICBU_P0715_2418066EC'  # 获取商品summary生成接口
RLAB_INSPIRATION_CONFIG = 'ICBU_N3984_245700899'  # Inspiration Config Extra信息查询工具

# Filter Suggestion相关
RLAB_PV_CTR = 'ICBU_P0715_2530233E6'  # CTR点击率工具
RLAB_WIKI_RAG = 'ICBU_N4552_25461339B'  # WIKI检索匹配Rag工具
RLAB_GOUTONG_RAG = 'ICBU_N4552_240214BA5'  # 沟通QA检索匹配Rag工具
RLAB_FILTER_CACHE = 'ICBU_P0715_259230738'  # FilterSuggestion缓存，用于缓存已有的数据
RLAB_CUSTOM_ATTR = 'ICBU_P0715_259067E94'  # FilterSuggestion对应的定制属性项
RLAB_CATE_SERVICE = 'ICBU_P0715_2420217F9'  # Cate大类对应的商家服务Service
RLAB_CATE_ASPECT = 'ICBU_P0715_260694515'  # Cate大类对应的Aspect维度


RLAB_PANEL_READ = 'ICBU_54410_161048B2C'  # 读取，并获取写权限
RLAB_PANEL_WRITE = 'ICBU_54410_161049F9C'  # 写，需要用获得写权限的userId才行

RLAB_CONFIG_CACHE = 'ICBU_N3984_255802E29'  # 通用缓存读取/写入

RLAB_PV_COUNT = 'ICBU_L5736_195616B62'  # 新需求 根据query返回PV count
RLAB_CERT_TOOLTIP = 'ICBU_L5736_2160015CE'  # 获取证书Tooltip

RLAB_BA_GLOBAL_SEARCH = 'ICBU_L5736_191081360'

# bge_reranker
# RLAB_BGE_RERANKER = 'ICBU_H1425_962013F4' # original reranker model
RLAB_BGE_RERANKER = 'ICBU_P0715_27341076F'

RERANKER_API_KEY = 'C5GHKY6HB8'

RLAB_BA_SEARCH_WITH_COST = 'ICBU_P0715_267808588'
RLAB_BA_COST_COMPARE = "ICBU_P0715_271623C1C"
RLAB_SEARCH_SUPPLIER_INFO = "ICBU_P0715_2736435B3"
RLAB_SEARCH_COMPANY_INFO_BY_ID = "ICBU_P0715_2836106A9"

RLAB_REQUEST_INSPIRATION = 'ICBU_N4004_27480189A'

RALB_RE_QUERY_REC = 'ICBU_P0715_25561687E'
RLAB_ACCIO_MEMORY = 'ICBU_P0715_2876036D2'
RLAB_SUPPLIER_INFO = 'ICBU_P0715_288407F17'
RLAB_PRODUCT_COMPARE = 'ICBU_P0715_2888070D6'
RLAB_RE_QUERY_REC = 'ICBU_P0715_25561687E'

# 调用sourcing Agent
RLAB_SOURCING_AGENT = "ICBU_183415_156804198"

RLAB_BIZ_ICBU_STATISTICS = 'ICBU_N4004_2798058AC'

RLAB_GOOGLE_TRENDS = 'ICBU_P0715_2786580FD'
RLAB_GOOGLE_TRENDS_NEW = 'ICBU_P0715_288406480'

RLAB_PRODUCT_DOCVIEW = 'ICBU_L5736_191291E6F'

RLAB_ENVIRON_2_URL_DICT = {
    'pre': 'https://pre-rlab.alibaba-inc.com/service/v1/open/',
    # 'pre': 'https://rlab.alibaba-inc.com/service/v1/open/',
    'online': 'https://rlab.alibaba-inc.com/service/v1/open/',
}

PROJECT_ENV = os.getenv('PROJECT_ENV', 'online')

tool_choice_cache_template = (
    'ACCIO#TOOL_CHOICE#ONLINE_CACHE#V1.1@{stripped_query_in_lower_case}'
)

RLAB_US_URL = 'https://us-rlab.alibaba-inc.com/service/v1/open/'
RLAB_TREND_IMAGE = "ICBU_N4004_279824DB9" # RLab for trending image

def parse_json(json_str):
    try:
        pattern = r'```(?:json)?\n(.*?)\n```'
        match = re.search(pattern, json_str, re.DOTALL)
        res = {}
        if match:
            # 解析提取的内容为 JSON 格式
            res = json.loads(match.group(1))
        elif is_json(json_str):
            res = json.loads(json_str)
        elif not json_str.endswith("\n```"):
            res = parse_json(json_str + "\n```")
        return res
    except Exception as e:
        print(e)
        return {}

class RlabApi(object):
    def __init__(self):
        self.rlab_environ = os.getenv('RLAB_ENVIRON', 'online')
        if PROJECT_ENV == 'online':
            self.rlab_url = 'http://rlab-service.accio.alibaba.vipserver/service/v1/open/'
        else:
            self.rlab_url = RLAB_ENVIRON_2_URL_DICT[self.rlab_environ]
        self.rlab_us_url = RLAB_US_URL

        # self.rlab_url = RLAB_ENVIRON_2_URL_DICT[self.rlab_environ]

    def request(self, data, inst_id, request_id='', agent_version='', timeout=20, region='CN'):
        try:
            headers = {'requestId': request_id}

            headers.update(RLAB_API_HEADERS)
            data['payload']['systemParams'] = data['payload'].get('systemParams', {})
            data['payload']['systemParams'].update({'ba_version': agent_version})
            if region == 'US':
                response = requests.post(
                    url=self.rlab_us_url + inst_id,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=timeout,
                )
            else:
                response = requests.post(
                    url=self.rlab_url + inst_id,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=timeout,
                )
            output = json.loads(response.text)
        except Exception as e:
            output = {}
            print(traceback.print_exc())
        return output

    def request_stream(self, data, inst_id, request_id='', agent_version='', timeout=20, region='CN'):
        try:
            headers = {'requestId': request_id}
            headers["rlab-response"] = "stream"
            # headers = {'requestId': request_id, 'version': agent_version}
            headers.update(RLAB_API_HEADERS)
            data['payload']['systemParams'] = data['payload'].get('systemParams', {})
            data['payload']['systemParams'].update({'ba_version': agent_version})
            if region == 'US':
                response = requests.post(
                    url=self.rlab_us_url + inst_id,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=timeout,
                    stream=True
                    )
            else:
                response = requests.post(
                    url=self.rlab_url + inst_id,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=timeout,
                    stream=True
                )

            def stream_generator():
                try:
                    for line in response.iter_lines():
                        if line:  # 过滤空行
                            decoded_line = line.decode('utf-8')
                            yield decoded_line  # 可根据需要解析JSON
                finally:
                    response.close()
            return stream_generator()
        except Exception as e:
            output = {}
        return output

    def fetch_navigator_inspiration_data(
        self, product_id_list, request_id='', agent_version=''
    ):
        try:
            request_data = {'payload': {'prod_ids': product_id_list}}
            results = self.request(
                request_data,
                inst_id=RLAB_ISPRTN_FOR_NVGT,
                request_id=request_id,
                agent_version=agent_version,
            )
            cluster_infos = (
                results.get('payload', {}).get('results', {}).get('clusterList', [])
            )
            if cluster_infos and isinstance(cluster_infos, list):
                result = {
                    k: v
                    for k, v in cluster_infos[0].items()
                    # if k
                    # in [
                    #     'score',
                    #     'estimatedGrossMargin',
                    #     'searchData',
                    #     'retailSalesData',
                    # ]
                }
            else:
                result = {}
        except Exception as e:
            result = {}
        return result

    def read_config_panel(self, session_id, request_id='', agent_version=''):
        if session_id:
            request_data = {'payload': {'sessionKey': session_id}}
            result = self.request(
                request_data,
                inst_id=RLAB_PANEL_READ,
                request_id=request_id,
                agent_version=agent_version,
            )  # 读取，并获取写锁
        else:
            print('Error! Should provide user_id when want to acquire lock!')
            result = {}
        return result

    def write_config_panel(self, session_id, config, request_id='', agent_version=''):
        if session_id:
            request_data = {
                'payload': {
                    'sessionKey': session_id,
                    'config': config,
                }
            }
            result = self.request(
                request_data,
                inst_id=RLAB_PANEL_WRITE,
                request_id=request_id,
                agent_version=agent_version,
            )  # 写，前提是用对应的userId拿到了锁
        else:
            print(
                'Error! Should provide session_id & config when want to write config panel!'
            )
            result = {}
        return result

    def retrieve_category_knowledge(self, data, request_id='', agent_version=''):
        res = self.request(
            data,
            inst_id=RLAB_KNOWLEDGE_ID,
            timeout=5,
            request_id=request_id,
            agent_version=agent_version,
        )
        return json.loads(
            res.get('payload', {}).get('content', {}).get('knowledge', '{}')
        )

    def retrieve_user_recent_queries(self, info, request_id='', agent_version=''):
        data = {'payload': info}
        try:
            res = self.request(
                data,
                inst_id=RLAB_USER_QUERIES_ID,
                timeout=5,
                request_id=request_id,
                agent_version=agent_version,
            )
            return res['payload']['query_history']
        except Exception as e:
            return ''

    def get_category_conversion_info(self, query, request_id='', agent_version=''):
        data = {'payload': {'query': query}}
        try:
            res = self.request(
                data,
                inst_id=RLAB_CATE_CVR_INFO,
                timeout=5,
                request_id=request_id,
                agent_version=agent_version,
            )
            return res['payload']['result']
        except Exception as e:
            return ''

    def retrieve_category_info(self, data, request_id='', agent_version=''):
        res = self.request(
            data,
            inst_id=RLAB_CATEINFO_ID,
            timeout=5,
            request_id=request_id,
            agent_version=agent_version,
        )
        return res['payload']['content']

    def category_prediction(self, data, request_id='', agent_version=''):
        categories = self.request(
            data, inst_id=RLAB_QP_ID, request_id=request_id, agent_version=agent_version
        )
        return categories

    def read_bing_search_cache(self, key, request_id='', agent_version=''):
        return self.general_rdb_cache_read(key, request_id, agent_version)

    def read_navigator_online_cache(self, key, request_id='', agent_version=''):
        return self.general_rdb_cache_read(key, request_id, agent_version)

    def read_navigator_offline_intervention(self, key, request_id='', agent_version=''):
        return self.general_rdb_cache_read(key, request_id, agent_version)

    def general_rdb_cache_read(self, key, request_id='', agent_version=''):
        data = {'payload': {'operation': 'read', 'key': key}}
        # print(json.dumps(data, indent=4))

        response = self.request(
            data,
            inst_id=RLAB_ONLINE_CACHE_ID,
            request_id=request_id,
            agent_version=agent_version,
        )
        try:
            cache = response.get('payload', {}).get('result', {}).get('value', '')
            cache_formatted = json.loads(cache)
        except Exception as e:
            cache_formatted = None
        return cache_formatted

    def write_bing_search_cache(
        self,
        key,
        value,
        expire_after_n_seconds=1 * 24 * 60 * 60,
        request_id='',
        agent_version='',
    ):
        self.general_rdb_cache_write(
            key, value, expire_after_n_seconds, request_id, agent_version
        )

    def write_navigator_online_cache(
        self,
        key,
        value,
        expire_after_n_seconds=1 * 24 * 60 * 60,
        request_id='',
        agent_version='',
    ):
        self.general_rdb_cache_write(
            key, value, expire_after_n_seconds, request_id, agent_version
        )

    def general_rdb_cache_write(
        self,
        key,
        value,
        expire_after_n_seconds=24 * 60 * 60,
        request_id='',
        agent_version='',
    ):
        data = {
            'payload': {
                'operation': 'write',
                'key': key,
                'value': json.dumps(value),
                'expire': expire_after_n_seconds,  # seconds as unit
                'regions': 'cn,us,sg',
            }
        }
        # print(json.dumps(data, indent=4))
        response = self.request(
            data,
            inst_id=RLAB_ONLINE_CACHE_ID,
            request_id=request_id,
            agent_version=agent_version,
        )
        return None

    def check_tool_choice_cache(self, query, request_id='', agent_version=''):
        key = tool_choice_cache_template.format(
            stripped_query_in_lower_case=query.strip().lower()
        )
        cache = self.general_rdb_cache_read(
            key, request_id + '-read_tool_choice', agent_version
        )

        try:
            if cache:
                return cache['tool_name']
            else:
                return None
        except Exception as e:
            return None

    def write_tool_choice_cache(
        self,
        query,
        tool_name,
        expire_after_n_seconds=7 * 24 * 60 * 60,
        request_id='',
        agent_version='',
    ):
        key = tool_choice_cache_template.format(
            stripped_query_in_lower_case=query.strip().lower()
        )
        self.general_rdb_cache_write(
            key,
            {'tool_name': tool_name},
            expire_after_n_seconds,
            request_id + '-write_tool_choice',
            agent_version,
        )

    def update_session_messages(
        self, messages, session_id, request_id='', agent_version=''
    ):
        messages_str = json.dumps(messages)
        data = {
            'payload': {
                'cacheKey': 'buyer_agent',
                'sessionKey': session_id,
                'value': messages_str,
            }
        }
        response = self.request(
            data,
            inst_id=RLAB_MSG_SAVE_ID,
            request_id=request_id,
            agent_version=agent_version,
        )

    def retrieve_session_messages(self, session_id, request_id='', agent_version=''):
        data = {'payload': {'cacheKey': 'buyer_agent', 'sessionKey': session_id}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_MSG_FETCH_ID,
                request_id=request_id,
                agent_version=agent_version,
            )
            return json.loads(results['payload']['result'])
        except Exception as e:
            return []

    def update_session_suggestions(
        self, suggestions, session_id, request_id='', agent_version=''
    ):
        suggestions_str = json.dumps(suggestions)
        data = {
            'payload': {
                'cacheKey': 'buyer_agent',
                'sessionKey': session_id + '_suggestions',
                'value': suggestions_str,
            }
        }
        response = self.request(
            data,
            inst_id=RLAB_MSG_SAVE_ID,
            request_id=request_id,
            agent_version=agent_version,
        )

    def retrieve_session_suggestions(self, session_id, request_id='', agent_version=''):
        data = {
            'payload': {
                'cacheKey': 'buyer_agent',
                'sessionKey': session_id + '_suggestions',
            }
        }
        try:
            results = self.request(
                data,
                inst_id=RLAB_MSG_FETCH_ID,
                request_id=request_id,
                agent_version=agent_version,
            )
            return json.loads(results['payload']['result'])
        except Exception as e:
            return []

    def get_prod_pv(self, prod_id_list, request_id='', agent_version=''):
        data = {'payload': {'prod_ids': prod_id_list}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_PROD_PV_READ,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['result']
        except Exception as e:
            return []

    def retrieve_pv_count(self, query, request_id='', agent_version=''):
        data = {'payload': {'query': query}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_PV_COUNT,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']
        except Exception as e:
            return {}

    def fetch_cert_tooltip(self, cert_names, request_id='', agent_version=''):
        data = {'payload': {'cert_names': cert_names}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_CERT_TOOLTIP,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['result']
        except Exception as e:
            return []

    def fetch_prod_info(self, prod_id_list, request_id='default', agent_version=''):
        data = {'payload': {'prod_ids': prod_id_list, 'traceId': request_id}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_PROD_INFO,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            return []

    def fetch_prod_basic_info(
        self, prod_id_list, request_id='default', agent_version=''
    ):
        ids_str = ', '.join([str(v) for v in prod_id_list])
        data = {'payload': {'ids': ids_str}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_PROD_INFO_BASIC,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['hsf_attr_table']['businessOfferList']
        except Exception as e:
            print(e)
            return []

    def fetch_prod_detail_info(
        self, prod_id_list, request_id='default', agent_version=''
    ):
        data = {'payload': {'prod_ids': prod_id_list, 'traceId': request_id}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_PROD_DETAIL,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            return []
        
    def manufacture_name_recognize(
        self, query, request_id='default', agent_version=''
    ):
        data = {'payload': {'query': query}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_MANUFACTURE_NAME_RECOGNIZE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['result']
        except Exception as e:
            return False

    def query_picture_match(
        self, query, images, prod_ids, request_id='default', agent_version=''
    ):
        data = {
            'payload': {
                'query': query,
                'images': images,
                'prod_ids': prod_ids,
                'traceId': request_id,
                'model': 'internvl2',  # internvl2/gpt-4o
            }
        }
        print('\npic_match_data', json.dumps(data, ensure_ascii=False))
        try:
            results = self.request(
                data,
                inst_id=RLAB_PICTURE_MATCH,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def picture_query_generation(self, images, request_id='default', agent_version=''):
        data = {
            'payload': {
                'images': images,
                'traceId': request_id,
                'model': 'gpt-4o',
                'language': 'EN',
            }
        }
        print('picture_query_generation')
        print(json.dumps(data, ensure_ascii=False))

        try:
            results = self.request(
                data,
                inst_id=RLAB_PICTURE_QUERYGEN,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def detail_query_generation(self, prod_id, request_id='default', agent_version=''):
        data = {
            'payload': {
                'sessionKey': '',
                'productId': prod_id,
                'language': 'EN',
                'locationId': '1',
            }
        }
        print('detail_query_generation')
        print(json.dumps(data, ensure_ascii=False))

        try:
            results = self.request(
                data,
                inst_id=RLAB_DETAIL_QUERYGEN,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['questions']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def compare_query_generation(
        self, prod_ids, request_id='default', agent_version=''
    ):
        data = {
            'payload': {
                'sessionKey': '',
                'productIds': prod_ids,
                'language': 'EN',
                'locationId': '1',
            }
        }
        print('compare_query_generation')
        print(json.dumps(data, ensure_ascii=False))

        try:
            results = self.request(
                data,
                inst_id=RLAB_COMPARE_QUERYGEN,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['questions']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def summary_generation(
        self, session_id, prod_id, request_id='default', agent_version=''
    ):
        data = {
            'payload': {
                'sessionKey': session_id,
                'productId': prod_id,
                'language': 'EN',
                'locationId': '1',
            }
        }
        # print('summary_generation')
        # print(json.dumps(data, ensure_ascii=False))

        try:
            results = self.request(
                data,
                inst_id=RLAB_SUMMARY_GEN,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def fetch_location_data(self, key, request_id='default', agent_version=''):
        data = {'payload': {'key': key}}
        print('fetch_location_data')
        print(json.dumps(data, ensure_ascii=False))

        try:
            results = self.request(
                data,
                inst_id=RLAB_LOCATION_DATA,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def fetch_inspiration_config(
        self,
        language: str,
        inspiration_demands: str,
        request_id='default',
        agent_version='',
    ):
        data = {
            'payload': {
                'language': language,
                'inspirationDemands': inspiration_demands,
            }
        }

        try:
            results = self.request(
                data,
                inst_id=RLAB_INSPIRATION_CONFIG,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['inspirationConfig']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def write_config_cache(
        self,
        key,
        value,
        expire_time=24 * 60 * 60,
        request_id='default',
        agent_version='',
    ):
        data = {
            'payload': {
                'operation': 'write',
                'key': key,
                'value': value,
                'expire': expire_time,
            }
        }
        try:
            results = self.request(
                data,
                inst_id=RLAB_CONFIG_CACHE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def read_config_cache(self, key, request_id='default', agent_version=''):
        if (
            self.rlab_environ != 'online' or PROJECT_ENV == 'local'
        ):  # 预发环境、本地环境关掉所有缓存
            return []

        data = {'payload': {'operation': 'read', 'key': key}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_CONFIG_CACHE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['result']['value']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def fetch_pv_ctr(self, cate_ids, request_id='default', agent_version=''):
        data = {'payload': {'cate_ids': cate_ids}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_PV_CTR,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def retrieve_goutong_qa(
        self, query, product_id, request_id='default', agent_version=''
    ):
        data = {'payload': {'query': query, 'tags': [str(product_id)]}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_GOUTONG_RAG,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['results']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def retrieve_wiki(self, query, section, request_id='default', agent_version=''):
        data = {'payload': {'query': query, 'tags': [section]}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_WIKI_RAG,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['retrieval']
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def write_filter_cache(
        self,
        key,
        value,
        expire_time=3 * 24 * 60 * 60,  # Filter缓存默认3天时间
        request_id='default',
        agent_version='',
    ):
        data = {
            'payload': {
                'operation': 'write',
                'key': key,
                'value': value,
                'expire': expire_time,
            }
        }
        try:
            results = self.request(
                data,
                inst_id=RLAB_FILTER_CACHE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results
        except Exception as e:
            import traceback

            traceback.print_exc()
            return []

    def read_filter_cache(self, key, request_id='default', agent_version=''):
        data = {'payload': {'operation': 'read', 'key': key}}
        try:
            results = self.request(
                data,
                inst_id=RLAB_FILTER_CACHE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['result']['value']
        except Exception as e:
            import traceback

    def fetch_custom_attri(self, cate_id, request_id='default', agent_version=''):
        data = {'payload': {'cate_id': cate_id, 'traceId': request_id}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_CUSTOM_ATTR,
                request_id=request_id,
                agent_version=agent_version,
            )
            return results['payload']['content']
        except Exception as e:
            return []

    def fetch_cate_service(
        self, cate_ids, language, request_id='default', agent_version=''
    ):
        data = {'payload': {'cate_ids': cate_ids, 'language': language}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_CATE_SERVICE,
                request_id=request_id,
                agent_version=agent_version,
            )
            return json.loads(results['payload']['results'])
        except Exception as e:
            return []

    def fetch_cate_aspect(self, cate_ids, request_id='default', agent_version=''):
        data = {'payload': {'cate_ids': cate_ids}}

        try:
            results = self.request(
                data,
                inst_id=RLAB_CATE_ASPECT,
                request_id=request_id,
                agent_version=agent_version,
            )
            return json.loads(results['payload']['results'])
        except Exception as e:
            return []

    def ba_search(self, param_data, request_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_BA_GLOBAL_SEARCH,
            request_id=request_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output

    def bge_reranker(self, query, docs, reranker_exp, request_id=""):
        print("Chunk Length:" + str(len(docs)), "Query:" + query)
        data = {
            "payload": {
                "api_key": RERANKER_API_KEY,
                "query": query,
                "contentList": docs
            }
        }
        timeout = 3
        if reranker_exp:
            data["payload"]["is_exp"] = True
            timeout = 5
        try:
            requests = self.request(
                data,
                inst_id=RLAB_BGE_RERANKER,
                request_id=request_id,
                timeout=timeout
            )
            return requests['payload']["data"]
        except Exception as e:
            import traceback
            print("bge_reranker fail to handle query:", query)
            traceback.print_exc()
            return {}

    def ba_search_with_cost(self, param_data, request_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_BA_SEARCH_WITH_COST,
            request_id=request_id,
            data=data,
            timeout=15,
            region='US'
        )
        rlab_output = response
        return rlab_output

    def ba_cost_compare(self, param_data, request_id='', region="CN"):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_BA_COST_COMPARE,
            request_id=request_id,
            data=data,
            timeout=15,
            region=region
        )
        rlab_output = response
        return rlab_output["payload"]

    def search_supplier_info(self, param_data, request_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_SEARCH_SUPPLIER_INFO,
            request_id=request_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output["payload"]

    def request_inspiration(self, query, request_id='', agent_version=''):
        data = {'payload': {'query': query}}
        response = self.request(
            inst_id=RLAB_REQUEST_INSPIRATION,
            request_id=request_id,
            data=data,
            agent_version=agent_version,
            region='US'

        )
        rlab_output = response
        return rlab_output["payload"]["result"]["clusterList"]
    
    def re_query_rec(self, intent_type_num, language, request_id='', agent_version=''):
        data = {'payload': {'intent_type_num': intent_type_num, "channel": "sourcingPlan", 'language': language}}
        response = self.request(
            inst_id=RLAB_RE_QUERY_REC,
            request_id=request_id,
            data=data,
            agent_version=agent_version
        )
        rlab_output = response
        return rlab_output["payload"]["result"]["query"]

    def search_company_info_by_id(self, param_data, request_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_SEARCH_COMPANY_INFO_BY_ID,
            request_id=request_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output["payload"]
    
    def operate_memory(self, param_data, requset_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_ACCIO_MEMORY,
            request_id=requset_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output["payload"]
        
    def request_trending_image_rlab(self, query, request_id='', agent_version=''):
        data = {'payload': {'query': query}}
        response = self.request(
            inst_id=RLAB_TREND_IMAGE,
            request_id=request_id,
            data=data,
            agent_version=agent_version
        )
        rlab_output = response
        return rlab_output["payload"]["image_urls"]

    def call_sourcing_agent(self, param_data, request_id=''):
        data = {'payload': param_data}
        response = self.request_stream(
            inst_id=RLAB_SOURCING_AGENT,
            request_id=request_id,
            data=data,
            timeout=100,
        )
        return response
    
    def get_supplier_info(self, param_data, requset_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_SUPPLIER_INFO,
            request_id=requset_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output["payload"]
    
    def product_compare(self, param_data, requset_id=''):
        data = {'payload': param_data}
        response = self.request(
            inst_id=RLAB_PRODUCT_COMPARE,
            request_id=requset_id,
            data=data,
            timeout=15,
        )
        rlab_output = response
        return rlab_output["payload"]


    def request_icbu_statistics(self, query, request_id='', agent_version=''):
        data = {
            'payload': 
            {
                "query_raw": query, 
                "apiKey": "ICBU_H1425_95682FCB-8E8D-4F2F-BFDD-86CD0FE87DD4C71",
                "systemParams": {
                    "ba_version": ""
                },
                "query": query,
                "appId": "ICBU_N4552_158606E84"
            }
        }
        response = self.request(
            inst_id=RLAB_BIZ_ICBU_STATISTICS,
            request_id=request_id,
            data=data,
            agent_version=agent_version,
        )
        rlab_output = response
        return rlab_output["payload"]["result"]

    def request_google_trends(self, query, method='', request_id='', agent_version=''):
        data = {'payload': {'query': query, "method": method}}
        response = self.request(
            inst_id=RLAB_GOOGLE_TRENDS_NEW,
            request_id=request_id,
            data=data,
            agent_version=agent_version,
            region='US'
        )
        rlab_output = response
        return rlab_output["payload"]["result"]
    
    def fetch_selected_prod_info(self, prod_id_list, request_id='default', agent_version=''):
        data = {'payload': {'prod_ids': prod_id_list, 'traceId': request_id}}
        results = self.request(
            data,
            inst_id=RLAB_PROD_INFO,
            request_id=request_id,
            agent_version=agent_version,
        )
        results = results['payload']['results']

        def remove_attributes(data):
            if isinstance(data, dict):
                new_data = {}
                for key, value in data.items():
                    if not any(x in key.lower() for x in ["class", "url", "id", "pic"]):
                        new_data[key] = remove_attributes(value)
                return new_data
            elif isinstance(data, list):
                return [remove_attributes(item) for item in data]
            else:
                return data
                
        return remove_attributes(results)
        




rlab_api = RlabApi()
