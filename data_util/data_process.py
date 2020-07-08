# coding=utf-8
import numpy as np
from tqdm import tqdm
import re
import jieba
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from data_util import config


class InputFeatures(object):
    def __init__(self, input_id, domain_label_id, input_mask, feature_list):
        self.input_id = input_id
        self.domain_label_id = domain_label_id
        self.input_mask = input_mask
        self.feature = feature_list


def read_corpus(path, max_length, intent2idx, slot2idx, vocab, is_train=True):
    """

    :param path: 数据地址
    :param max_length:句子最大长度
    :param intent2idx: 意图标签字典
    :param slot2idx: 槽位标签字典
    :param vocab: word 字典

    :return:
    """
    char2idx = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12,"m":13,"n":14,
                "o":15,"p":16,"q":17,"r":18,"s":19,"t":20,"u":21,"v":22,"w":23,"x":24,"y":25,"z":26,"'":27,"unk":28}
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    token_lists, slot_lists, intent_lists, mask_lists = [],[],[],[]
    char_lists=[]
    slot_outs=[]
    token_list, slot_list = [], []
    slot_out = []
    over_length = 0
    query_list = []
    max_len_word =0
    for idx,line in enumerate(content):
        line = line.strip()
        if line !="":
            line = line.split(" ")
            if len(line) ==1:
                intent = line[0]
                intent_lists.append(intent2idx[intent])
            if len(line)==2:
                token, slot = line[0],line[1]
                max_len_word = max(max_len_word, len(token))
                if token not in vocab:
                    token_list.append(vocab["<unk>"])
                else:
                    token_list.append(vocab[token])
                query_list.append(token)
                slot_list.append(slot2idx[slot])
                slot_out.append(slot)
        else:

            if len(token_list) > max_length-2:
                token_list = token_list[0 : (max_length - 2)]
                query_list = query_list[0 : (max_length - 2)]
                over_length+=1
            slot_list = slot_list[0: (max_length - 2)]
            slot_outs.append(slot_out)

            char_list=[]
            for token in query_list:
                chars = [char2idx[c] if c in char2idx else char2idx["unk"] for c in list(token)]
                if len(chars)<25:
                    chars= chars + (25-len(token))*[0]
                else:
                    chars=chars[:25]
                char_list.append(chars)
            char_list = [25*[0]] + char_list+[25*[0]]
            token_list = [vocab["</s>"]] + token_list + [vocab["</e>"]]
            slot_list = [slot2idx["<start>"]] + slot_list + [slot2idx["<end>"]]
            mask_list = [1] * len(token_list)
            while len(token_list) < max_length:
                char_list.append(25*[0])
                token_list.append(0)
                slot_list.append(slot2idx["<PAD>"])
                mask_list.append(0)
            assert len(token_list)==max_length and len(slot_list) == max_length and len(mask_list)==max_length
            assert len(char_list)==max_length
            token_lists.append(token_list)
            slot_lists.append(slot_list)
            mask_lists.append(mask_list)
            char_lists.append(char_list)
            query_list = []
            token_list, slot_list,slot_out= [], [],[]
    data_loader = toTensor(token_lists,char_lists, slot_lists,intent_lists, mask_lists,is_train=is_train)
    print("超过最大长度的样本数量为：",over_length)
    print(max_len_word)
    return data_loader


def toTensor(token_lists, char_lists, slot_lists,intent_lists, mask_lists,is_train=True):

    dataset = TensorDataset(torch.LongTensor(token_lists),torch.LongTensor(char_lists),torch.LongTensor(slot_lists),torch.LongTensor(intent_lists),torch.LongTensor(mask_lists))
    if is_train:
        data_loader = DataLoader(dataset, shuffle=True, batch_size = config.batch_size)
    else:
        data_loader = DataLoader(dataset, shuffle=False, batch_size=config.batch_size)
    return data_loader


def make_label(query, domain):
    domain_label = domain
    token_list =[]
    for token in query:
        token_list.append(token)
    return token_list, domain_label


def make_feature(query, mall_tag):
    feature_list =[0]*len(query)
    for start in range(len(query)):
        for end in range(start+1,len(query)+1):
            if query[start:end] in mall_tag:
                feature_list[start:end] = [1]*(end-start)

    return feature_list


def build_vocab(file_path, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in jieba_cut(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1

        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)
        vocab_list = [word_count[0] for word_count in vocab_list]
    return vocab_list


def read_emb (file_path, vocab_list ):
    embeddings=[]
    with open(file_path, 'r', encoding='UTF-8') as f :
        for emb in tqdm(f, "select_emb") :
            emb_list = emb.strip().split(" ")
            if emb_list[0] in vocab_list:
                embeddings.append(emb)

    return embeddings


def process_emb(embedding,emb_dim):
    embeddings = {}
    embeddings["<pad>"] = np.zeros(emb_dim)
    embeddings["<unk>"] = np.random.uniform(-0.01,0.01,size = emb_dim)
    embeddings["</s>"] = np.random.uniform(-0.01,0.01,size = emb_dim)
    embeddings["</e>"] = np.random.uniform(-0.01,0.01,size = emb_dim)

    for emb in embedding:
        line = emb.strip().split()
        word = line[0]
        word_emb = np.array([float(_) for _ in line[1:]])
        embeddings[word] = word_emb

    vocab_list = list(embeddings.keys())
    word2id ={vocab_list[i]:i for i in range(len(vocab_list))}
    embedding_matrix = np.array(list(embeddings.values()))

    return  embedding_matrix, word2id


def lord_label_dict(path):
    label2id = {}
    id2label = {}
    f=open(path, "r", encoding="utf-8")
    for item in f:
        id, label = item.strip().split("\t")
        label2id[label] = int(id)
        id2label[int(id)] = label
    f.close()
    return id2label, label2id


def jieba_cut(sen):
    sen_string = sen.strip()
    sen_string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[∠～【】¯╮╯▽╰︶⊙+——<⑅ↁ́ᴗↁ́⑅)“”：;｡◕‿◕｡！?，。？、~@#￥%……&*（）]", "",
                        sen_string)
    sen_list = jieba.cut(sen_string)
    return sen_list