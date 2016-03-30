#!/usr/bin/env python
#-*- coding: utf-8 -*-

from _srilm import *
import sys
import os

reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append("./config")

open_flag = [1, 0, 0, 1, 1, 1];

try:
    import sofa
except:
    sys.stderr.write('Error: Please excute the following command first:\n')
    sys.stderr.write('export SOFA_CONFIG=./config/drpc_client.xml\n')
    sys.exit(1)
sofa.use('drpc.ver_1_0_0', 'S')
sofa.use('nlpc.ver_1_0_0', 'nlpc')
conf = sofa.Config()
conf.load('./config/drpc_client.xml')#local


if open_flag[0] == 1:
    wordrank_agent = S.ClientAgent(conf['sofa.service.nlpc_wordrank_208'])#local
else:
    wordrank_agent = None;
if open_flag[1] == 1:
    wordpos_agent = S.ClientAgent(conf['sofa.service.nlpc_wordpos_202'])  #local
else:
    wordpos_agent = None;
if open_flag[2] == 1:
    depparser_agent = S.ClientAgent(conf['sofa.service.nlpc_depparser_query_107']) #local
else:
    depparser_agent = None;
if open_flag[3] == 1:
    lmscore_agent = S.ClientAgent(conf['sofa.service.nlpc_lmscore_1040'])
else:
    lmscore_agent = None;

if open_flag[4] == 1:
    wordner_agent = S.ClientAgent(conf['sofa.service.nlpc_wordner_300'])
else:
    wordner_agent = None;

if open_flag[4] == 1:
    no_gram = initLM(3);
    readLM(no_gram, "./lm/all_phase.0326.lm");
else:
    no_gram = -1;


def get_token_list_lm(token_list_all):
    if no_gram == -1:
        return -1, 0.0, 0.0
    token_list = [];
    for i in token_list_all:
        if i.strip() != "":
            token_list.append(i);
    str_new = " ".join(token_list);
    token_list_no = len(token_list);
    corpus_prob = getSentenceProb(no_gram, str_new, token_list_no)
    sppl = getSentencePpl(no_gram, str_new, token_list_no)
    noov = numOOVs(no_gram, str_new, token_list_no);
    return noov,corpus_prob,sppl



g_pos_tag_set = ["Ag","Dg","Ng","Tg","Vg","a","ad","an","b","c","d",
                "e","f","g","h","i","j","k","l","m","n","nr","ns",
                "nt","nx","nz","o","p","q","r","s","t","u","v","vd",
                "vn","w","y","z"]

def utf8_gbk(string):
    return string.decode('utf8','ignore').encode('gbk','ignore')

def gbk_utf8(string):
    return string.decode('gbk','ignore').encode('utf8','ignore')

def get_pos_str(nPos):
    if nPos <= 0 or nPos > 39:
        return 0
    else:
        return g_pos_tag_set[nPos - 1]

def word_pos(sentence):
    if not wordpos_agent:
        return [];
    ## post request
    m_input = nlpc.wordseg_input()
    m_input.lang_id = int(0)
    m_input.lang_para = int(0)
    m_input.query = str(utf8_gbk(sentence))
    input_data = sofa.serialize(m_input)
    for i in range(5) :
        try:
            ret, output_data = wordpos_agent.call_method(input_data)
            break
        except Exception as e:
            pass;
    if len(output_data) == 0:
        sys.stderr.write('No result' + sentence + '\n')
        return []
    
    ## get results
    # wordpos result
    m_output = nlpc.wordpos_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    tokens_size = len(m_output.nlpc_tokens)
    segment_result = []
    for i in range(tokens_size):
        stag = get_pos_str(m_output.nlpc_tokens[i].type)
        if stag:
            word = m_output.nlpc_tokens[i].buffer
            word = gbk_utf8(word)
            segment_result.append((word, stag))
    return segment_result

def word_rank(sentence):
    if not wordrank_agent:
        return [];
    ## post request
    m_input = nlpc.wordseg_input()
    m_input.lang_id = int(0)
    m_input.lang_para = int(0)
    m_input.query = str(utf8_gbk(sentence))
    input_data = sofa.serialize(m_input)
    for i in range(5) :
        try:
            ret, output_data = wordrank_agent.call_method(input_data)
            break
        except Exception as e:
            pass;
    if len(output_data) == 0:
        sys.stderr.write('No result' + sentence + '\n')
        return []
    
    ## get results
    # wordrank_result
    m_output = nlpc.wordrank_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    rank_result_list = list()
    list_size = len(m_output.nlpc_trunks_pn)
    for i in range(list_size):
        word = m_output.nlpc_trunks_pn[i].buffer
        word = gbk_utf8(word)
        rank = m_output.nlpc_trunks_pn[i].rank
        wght = round(m_output.nlpc_trunks_pn[i].weight,3)
        rank_result_list.append((word, rank, wght))
    return rank_result_list

def word_lmscore(sentence):
    if not lmscore_agent:
        return 0.0;
    m_input = nlpc.lmscore_input()
    m_input.query = str(utf8_gbk(sentence));
    m_input.debug_flag = True
    input_data = sofa.serialize(m_input)
    for i in range(5):
        try:
            ret, output_data = lmscore_agent.call_method(input_data)
            break
        except Exception as e:
            continue
    if len(output_data) == 0:
        return 0.0

    m_output = nlpc.lmscore_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    return m_output.result.prob;

trans_id_short = {};
trans_id_short[0] = "NOR";
trans_id_short[1] = "PER";
trans_id_short[2] = "LOC"
trans_id_short[3] = "ORG"
trans_id_short[4] = "SFT"
trans_id_short[5] = "GME"
trans_id_short[6] = "SNG"
trans_id_short[7] = "NVL"
trans_id_short[8] = "VDO"
trans_id_short[9] = "STE"
trans_id_short[10] = "BRD"
trans_id_short[11] = "BRD_ORG"
trans_id_short[12] = "CTN"
trans_id_short[13] = "MDL"
trans_id_short[14] = "PDT"
trans_id_short[15] = "PHRASE"
trans_id_short[16] = "ROOT"
trans_id_short[17] = "BRAN"
trans_id_short[18] = "CHL"
trans_id_short[19] = "STE_CRE"
trans_id_short[20] = "ATT"
trans_id_short[21] = "ATT_VDO"
trans_id_short[22] = "ATT_NVL"
trans_id_short[23] = "ATT_SFT"
trans_id_short[24] = "DYN_UNK"
trans_id_short[25] = "VDO_SUB"
trans_id_short[26] = "ORG_SFX"
trans_id_short[27] = "ORG_CRE"
trans_id_short[28] = "ORG_MOD"
trans_id_short[29] = "DYN_ORG"
trans_id_short[30] = "DYN_PER"
trans_id_short[31] = "UNK"
trans_id_short[32] = "MULTI"
trans_id_short[50] = "ILL"
trans_id_short[81] = "VDO_MVE"
trans_id_short[82] = "VDO_TV"
trans_id_short[83] = "VDO_TVSHOW"
trans_id_short[90] = "LOC_PRO"
trans_id_short[91] = "LOC_CIT"
trans_id_short[92] = "LOC_DIS"
trans_id_short[93] = "LOC_BLK"
trans_id_short[94] = "LOC_BUILDING"
trans_id_short[95] = "NER"
trans_id_short[101] = "AC"
trans_id_short[102] = "TV"
trans_id_short[103] = "MOV"
trans_id_short[104] = "SCI"
trans_id_short[201] = "RQST_PER"
trans_id_short[202] = "RQST_LOC"
trans_id_short[203] = "RQST_ORG"
trans_id_short[204] = "RQST_SFT"
trans_id_short[205] = "RQST_GME"
trans_id_short[206] = "RQST_SNG"
trans_id_short[207] = "RQST_NVL"
trans_id_short[208] = "RQST_VDO"
trans_id_short[209] = "RQST_STE"
trans_id_short[210] = "RQST_BRD"
trans_id_short[212] = "RQST_CTN"
trans_id_short[214] = "RQST_PDT"
trans_id_short[281] = "RQST_VDO_MVE"
trans_id_short[282] = "RQST_VDO_TV"
trans_id_short[283] = "RQST_VDO_TVSHOW"
trans_id_short[243] = "RQST_IMG"
trans_id_short[244] = "RQST_DOC"
trans_id_short[301] = "RQST_AC"
trans_id_short[302] = "RQST_TV"
trans_id_short[303] = "RQST_MOV"
trans_id_short[304] = "RQST_SCI"
trans_id_short[399] = "RQST"
trans_id_short[400] = "INVALID"

def word_ner(sentence):
    if not wordner_agent:
        return [];
    language_id = 0
    output_id = 1
    m_input = nlpc.wordner_input()
    m_input.lang_id = int(1)
    m_input.query = str(utf8_gbk(sentence))
    input_data = sofa.serialize(m_input)
    for i in range(5):
        try:
            ret, output_data = wordner_agent.call_method(input_data)
            break
        except Exception as e:
            continue
    if len(output_data) == 0:
        sys.stderr.write('The server returns None.' + '\n')
        return [];
    m_output = nlpc.wordner_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    tags = m_output.tags
    tags_size = len(tags)
    word_ner_list = [];
    for i in range(tags_size):
        word_ner_list.append((gbk_utf8(tags[i].term), str(tags[i].type), trans_id_short[tags[i].type]));
        '''
        sys.stderr.write(gbk_utf8(tags[i].term) + ' ')
        if trans_id_short.has_key(tags[i].type):
            sys.stderr.write(trans_id_short[tags[i].type] + '\t')
        else:
            sys.stderr.write('NOR' + '\t')
    sys.stderr.write('\n')
    '''
    return word_ner_list;



def word_depparser(sentence, is_segmented=False):
    if not depparser_agent:
        return [];
    ## post request
    m_input = nlpc.parse_prep_input()
    m_input.sentence = str(utf8_gbk(sentence))
    m_input.grain_size = 1 
    m_input.sentence_segmented = is_segmented
    input_data = sofa.serialize(m_input)
    for i in range(5) :
        try:
            ret, output_data = depparser_agent.call_method(input_data)
            break
        except Exception as e:
            continue
    if len(output_data) == 0:
        sys.stderr.write('No result' + sentence + '\n')
        return []
    
    ## get results
    m_output = nlpc.depparser_output()
    m_output = sofa.deserialize(output_data, type(m_output))
    tokens = m_output.items
    depparser_list = []
    for i in range(len(tokens)):
        if len(tokens[i].deprel.strip()) == 0:
            tokens[i].deprel = '_'
        word = gbk_utf8(tokens[i].word)
        depparser_list.append((word, tokens[i].deprel))
    return depparser_list

if __name__ == '__main__':
 
    '''
    print '***** wordner *******'
    word_ner_list = word_ner('刘德华')
    for item in word_ner_list:
        print  ":".join(list(item));

    sentence = '百度是全球第一的搜索引擎！李彦宏'
           

    print '****** pos *******'
    pos_list = word_pos(sentence)
    for item in pos_list:
        print ':'.join(list(item))
    
    print '****** rank *******'
    rank_list = word_rank(sentence)
    for item in rank_list:
        print ':'.join(list([str(val) for val in item]))

    term_list = [];
    for i in rank_list:
        term_list.append(i[0].encode('UTF-8', 'ignore'));

    print '****** depparser *******'
    dep_list = word_depparser(sentence)
    for item in dep_list:
        print ':'.join(list(item))

    print '***** lmscore *******';
    print " ".join(term_list);
    print word_lmscore(" ".join(term_list));

    for line in sys.stdin:
        line = line.strip()
        dep_list = word_depparser(line)
        print line + '\t' + ' '.join(['*#*'.join(list(dep)) for dep in dep_list])
    '''
