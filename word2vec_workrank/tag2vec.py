#!usr/bin/env python
#-*- coding: utf-8 -*-

import sys
import os

import my_nlpc_tools as nlpc_tools
import pymongo
import my_python_tools as py_tools

reload(sys)
sys.setdefaultencoding('utf8')

USE_FILE = False
if USE_FILE == True:
    # load from file
    word_vecs = py_tools.load_kv_file('/home/disk2/zouxiaoyi/word2vec/douban_movie_vec.norm', 0, 1, True)
else:
    # connect the database
    mongodb = pymongo.MongoClient("localhost", 18888)
    wordembd_collection = mongodb['dumi']['query_feat']

def get_term_feat(term):
    try:
        if USE_FILE == True:
            term = py_tools.unicode_encode(term)
            feat = word_vecs.get(term, '')
        else:
            item = wordembd_collection.find_one({'_id':term})
            feat = item.get('feat', '')
        feat = [float(val) for val in feat.split()]
    except:
        feat = []
    return feat

def get_sum_feat(word_rank_weight):
    sum_feat = []
    for term in word_rank_weight:
        word = term[0]
        weight = float(term[1]) * float(term[2])
        feat = get_term_feat(word)
        if len(feat) == 0:
            continue
        if len(sum_feat) == 0:
            sum_feat = feat
        else:
            sum_feat = py_tools.add_two_list(sum_feat, feat, weight)
    return sum_feat

def tag2vec(tag):
    word_rank_weight = nlpc_tools.word_rank(tag)
    if len(word_rank_weight) == 0:
        sys.stderr.write('Warning: no word rank result: ' + line)
        return []
    del(word_rank_weight[-1])
    return get_sum_feat(word_rank_weight)

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        vec = tag2vec(line)
        if len(vec) > 0:
            print line + '\t' + ' '.join([str(round(f, 3)) for f in vec])
