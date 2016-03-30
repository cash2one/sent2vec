#!/usr/bin/env python
#-*- coding:UTF-8 -*-

import sys;
import re;
import pymongo
import time;
import json;
from bson import json_util
import urllib 
reload(sys);
sys.setdefaultencoding('UTF-8');

ISOTIMEFORMAT="%Y-%m-%d %X"
conn = pymongo.MongoClient("127.0.0.1",18888)


def get_format_baike_item(token_list):
    data_ret = {};
    schema = ["new_id", "name", "lemma_desc", "citiao_id", "yixiang_id", "daily_pv", "entity_category", "baike_url_1", "baike_url_2"];
    idx = 0;
    for i in schema:
        data_ret[schema[idx]] = token_list[idx];
        idx = idx + 1;
    return data_ret;

'''
for line in sys.stdin:
    token_list = line.rstrip().split('\t');
    if len(token_list) != 9:
        print >> sys.stderr,'unformat line-->', line.rstrip();
        continue;
    data_item = get_format_baike_item(token_list);
    db.meta_data.insert(data_item) 
'''
#content = db.meta_data.find()
#for i in content:
#    print i

###create TABLE;
'''
db = conn.sys_info
db = conn.sys_info
db.table_info.insert({'dataset_class':"spo_data", "dataset_name":"人物关系_0806", "time": time.strftime( ISOTIMEFORMAT, time.localtime()), "user": "wangbo01", "lineno": 123, "status":"OK", "description":"人物关系数据"})

content = list(db.table_info.find());
content = json.dumps(content,  default=json_util.default);
print content;
exit(0)
'''

##insert data;
db = conn.dumi;
db.query_feat.remove()

idx = 0;
for line in sys.stdin:
    idx = idx + 1;
    if idx == 1:
        continue;
    token_list = line.rstrip().split(' ');
    data_ret = {};
    if len(token_list) != 201:
        print line.strip() + "\t" + str(len(token_list));
        exit(-1);
    data_ret["_id"] = token_list[0];
    data_ret["feat"] = " ".join(token_list[1:]);
    try:
        db.query_feat.insert(data_ret);
    except Exception, e:
        print "error add idx:" + str(idx)
        continue;
    print "success add idx:" + str(idx)
'''
###select data
db = conn.dumi;
content = list(db.query_feat.find());
print content;
'''
