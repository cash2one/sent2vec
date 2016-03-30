#!/usr/bin/env python
#-*- coding:UTF-8 -*-

## 伪代码

word_embeding_dict = {};
idx = 0;
for line in open("dict/word_embedding"):
    token_list = line.strip().split('\t');
    word = token_list[0];
    feat = get_term_feat(word, VEC_SIZE);
    word_embeding_dict[word] = feat;
    idx = idx + 1;
    if idx >= EMBEDDING_SIZE:
        break;
logging.info('load word_embedding: '+ str(len(word_embeding_dict)));

def get_word_embedding_feat(word_rank_weight, vec_size):
    term_feat = [];
    for term in word_rank_weight:
        word = term[0];
        feat = get_term_feat(word, vec_size)
        term_feat.append(feat);

    sum_feat = []
    for k,v in word_embeding_dict.items():
        vec = v;
        max_sim = 0.0;
        for vec_term in term_feat:
            sim_c = py_tools.cosine_similarity(vec_term, vec);
            if sim_c > max_sim:
                max_sim = sim_c;
        sum_feat.append(max_sim);
    return sum_feat


    for i in word_rank_weight:
        ret_list.append(i[0].encode('UTF-8', 'ignore'));
    return ret_list;

vec = get_word_embedding_feat(word_rank_weight, VEC_SIZE);
vec_str = [str(round(f, 3)) for f in vec];
print vec_str;
