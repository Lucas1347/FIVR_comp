#!/usr/bin/env python
# coding: utf-8

# # [Fine-grained Incident Video Retrieval (FIVR)](https://arxiv.org/abs/1809.04094)
# ## Dataset
# - 数据集特征目录：/home/camp/FIVR/features/vcms_v1
# - Annotation目录：/home/camp/FIVR/annotation
# - 描述：我们提供视频的帧级别特征
#     - 每个h5文件有两个group: images和names
#     - images group 保存了每个视频帧的特征，id是视频id，value是帧特征
#     - names group 保存了每个视频帧的名字，id是视频id，value是帧的名字，例如\[1.jpg, 2.jpg,...\]
# - 一些关键词的解释
#     - vid: 和视频一一对应。
#     - name: 和视频一一对应，annotation中的使用的是name
#     - 通过vid2name和name2vid可以确定他们之间的映射关系
# - 三种相似的视频
#     - Duplicate Scene Video (DSV)
#     - Complementary Scene Video (CSV), 如果A，B两个视频描述的同一个事件，且时间上有overlap，则认为是彼此之间的相似关系是CSV
#     - Incident Scene Video (ISV)，如果A，B两个视频描述的是同一个时间，时间上没有overlap，则认为彼此之间的相似关系是ISV
# - 三个任务
#     - DSVR：负责检索出DSV的相似
#     - CSVR：负责检索出DSV+CSV的相似
#     - ISVR：负责检索出DSV+CSV+ISV的相似

# In[1]:


import numpy as np
import os
import h5py
from tqdm import tqdm
from glob import glob
import pickle as pk
import json
import time
from scipy.spatial.distance import cdist
from future.utils import viewitems, lrange
from sklearn.metrics import precision_recall_curve
from multiprocessing import Process
from multiprocessing import Pool
# In[2]:


def read_h5file(path):
    hf = h5py.File(path, 'r')
    g1 = hf.get('images')
    g2 = hf.get('names')
    return g1.keys(), g1, g2
def load_features(dataset_dir, is_gv=True):
    '''
    加载特征
    :param dataset_dir: 特征所在的目录, 例如：/home/camp/FIVR/features/vcms_v1
    :param is_gv: 是否取平均。True：返回帧平均的结果，False：保留所有帧的特征
    :return:
    '''
    h5_paths = glob(os.path.join(dataset_dir, '*.h5'))
    print(h5_paths)
    vid2features = {}
    final_vids = []
    features = []
    for h5_path in h5_paths:
        vids, g1, g2 = read_h5file(h5_path)
        for vid in tqdm(vids):
            if is_gv:
                cur_arr = g1.get(vid)
                #print("1:",cur_arr.shape)
                cur_arr_ave = np.mean(cur_arr, axis=0, keepdims=False)
                cur_arr_max = np.max(cur_arr, axis=0, keepdims=False)
                cur_arr = np.concatenate([cur_arr_ave, cur_arr_max], axis=0)
                cur_arr /= (np.linalg.norm(cur_arr, ord=2, axis=0))
                #print(cur_arr.shape)
                vid2features[vid] = cur_arr
            else:
                cur_arr = g1.get(vid)
                #print("1:",cur_arr.shape)
                cur_arr = np.concatenate([cur_arr], axis=0)
                vid2features[vid] = cur_arr
                #print(cur_arr.shape)
                final_vids.extend([vid] * len(cur_arr))
                features.extend(cur_arr)
    if is_gv:
        return vid2features
    else:
        return final_vids, features, vid2features

def calculate_similarities(query_features, all_features, w):
    """
      用于计算两个视频之间的累计相似度，带权
      Args:
        queries: shape: [N, D]
        features: shape: [M, D]
      Returns:
        similarities: float
    """
    similarities = 0.0
    # 计算待查询视频和所有视频的距离
    dist = np.nan_to_num(cdist(query_features, all_features, metric='cosine'))
    sim = 1 - dist
    sim = np.max(sim, axis=1)
    similarities = np.sum(sim * w)
    return similarities


def evaluateOfficial(annotations, results, relevant_labels, dataset, quiet):
    """
      Calculate of mAP and interpolated PR-curve based on the FIVR evaluation process.
      Args:
        annotations: the annotation labels for each query
        results: the similarities of each query with the videos in the dataset
        relevant_labels: labels that are considered positives
        dataset: video ids contained in the dataset
      Returns:
        mAP: the mean Average Precision
        ps_curve: the values of the PR-curve
    """
    pr, mAP = [], []
    iterations = viewitems(annotations) if not quiet else tqdm(viewitems(annotations))
    for query, gt_sets in iterations:
        query = str(query)
        if query not in results: print('WARNING: Query {} is missing from the result file'.format(query)); continue
        if query not in dataset: print('WARNING: Query {} is not in the dataset'.format(query)); continue

        # set of relevant videos
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(dataset)
        if not query_gt: print('WARNING: Empty annotation set for query {}'.format(query)); continue

        # calculation of mean Average Precision (Eq. 6)
        i, ri, s = 0.0, 0, 0.0
        y_target, y_score = [], []
        for video, sim in sorted(viewitems(results[query]), key=lambda x: x[1], reverse=True):
            if video in dataset:
                y_score.append(sim)
                y_target.append(1.0 if video in query_gt else 0.0)
                ri += 1
                if video in query_gt:
                    i += 1.0
                    s += i / ri
        mAP.append(s / len(query_gt))
        #if not quiet:
        #    print('Query:{}\t\tAP={:.4f}'.format(query, s / len(query_gt)))

        # add the dataset videos that are missing from the result file
        missing = len(query_gt) - y_target.count(1)
        y_target += [1.0 for _ in lrange(missing)] # add 1. for the relevant videos
        y_target += [0.0 for _ in lrange(len(dataset) - len(y_target))] # add 0. for the irrelevant videos
        y_score += [0.0 for _ in lrange(len(dataset) - len(y_score))]

        # calculation of interpolate PR-curve (Eq. 5)
        precision, recall, thresholds = precision_recall_curve(y_target, y_score)
        p = []
        for i in lrange(20, -1, -1):
            idx = np.where((recall >= i * 0.05))[0]
            p.append(np.max(precision[idx]))
        pr.append(p)
    # return mAP
    return mAP, np.mean(pr, axis=0)[::-1]

class GTOBJ:
    def __init__(self):
        annotation_path = '/home/camp/FIVR/annotation/annotation.json'
        dataset_path = '/home/camp/FIVR/annotation/youtube_ids.txt'
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        self.dataset = set(np.loadtxt(dataset_path, dtype=str).tolist())
gtobj = GTOBJ()
relevant_labels_mapping = {
    'DSVR': ['ND','DS'],
    'CSVR': ['ND','DS','CS'],
    'ISVR': ['ND','DS','CS','IS'],
}


# In[3]:


tem_, tem__, vid2features = load_features('/home/camp/FIVR/features/vcms_v1', is_gv=False)


# In[4]:


# 加载特征
vids = list(vid2features.keys())
print(vids[:10])
#print(list(vid2features.values())[:2])

# global_features = np.squeeze(np.asarray(list(vid2features.values()), np.float32))
global_feattures = [np.asarray(i,np.float32) for i in list(vid2features.values())]
# print(np.shape(global_features))


# In[5]:


# 加载vid2name 和 name2vid
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    vid2names = pk.load(pk_file)
with open('/home/camp/FIVR/vid2name.pk', 'rb') as pk_file:
    name2vids = pk.load(pk_file)


# In[6]:

def self_sim_matrix(feat):
    """
       计算自相关矩阵，用来self_attention
    """
    dist = np.nan_to_num(cdist(feat, feat, metric='cosine'))
    sim = 1 - dist
    sim = np.sum(sim, axis=1, keepdims=False)
    return sim


def query_sim(arg):
    name, id = arg
    similarities = dict()
    query_features = global_feattures[id]
            
    self_attention_w = np.ones(query_features.shape[0])
    self_attention_w = self_sim_matrix(query_features)    
    
    for __,temp_feature in enumerate(global_feattures):
        now_similarities = calculate_similarities(query_features, temp_feature, self_attention_w)
        similarities[names[__]] = now_similarities
    similarities = dict(sorted(similarities.items(),key = lambda k:k[1]))
    del similarities[name]
    
    print(name, " done.")
    return (name, similarities)


# 开始评估
annotation_dir = '/home/camp/FIVR/annotation'
names = np.asarray([vid2names[vid][0] for vid in vids])
query_names = None
results = None
for task_name in ['DSVR', 'CSVR', 'ISVR']:
    annotation_path = os.path.join(annotation_dir, task_name + '.json')
    with open(annotation_path, 'r') as annotation_file:
        json_obj = json.load(annotation_file)
    if results is None:
        pool=Pool(processes=50)

        query_names = json_obj.keys()
        query_names = [str(query_name) for query_name in query_names]
        query_indexs = []
        print("query len:", len(query_names))
        for query_name in query_names:
            tmp = np.where(names == query_name)
            if len(tmp) != 0 and len(tmp[0]) != 0:
                query_indexs.append(tmp[0][0])
            else:
                print('skip query: ', query_name)
        # print(len(query_indexs),query_indexs[0])
        results = dict()

        args = []
        for _,id in enumerate(query_indexs):
            args.append((query_names[_],id))

        res = pool.map(query_sim, args)
        print("len(res):", len(res))
        for itm in res:
            name, sim = itm
            results[name] = sim

        print("len(res):", len(results))
    mAPOffcial, precisions = evaluateOfficial(annotations=gtobj.annotations, results=results,
                                                  relevant_labels=relevant_labels_mapping[task_name],
                                                  dataset=gtobj.dataset,
                                                  quiet=False)
    print('{} mAPOffcial is {}'.format(task_name, np.mean(mAPOffcial)))


# In[ ]:
