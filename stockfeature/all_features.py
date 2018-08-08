#-*-encoding=utf-8-*-

from __future__ import print_function
import os
import pprint
import json
import numpy as np
#import talib
from feature_factory import Feature_Factory
#from talib.abstract import *
#import talib

'''
Fit all features into file
'''

class All_Features(object):
    '''
    Read raw stock data from file
    and map into json to file
    '''
    def __init__(self,paths,conf,extra_conf=None,type=None):
        self.paths = paths if paths.__class__ == list else [paths]
        self.type = type
        self.conf = conf
        self.feat_conf = None
        self.feat_extra_conf = None
        try:
            self.feat_conf = open(conf).readline().strip().split()
        except:
            print('Conf File Not Valid')
        try:
            self.feat_extra_conf = open(extra_conf).readline().strip().split()
        except:
            print('Extra Conf File Not Valid')
    
    def map_feat(self,vecs,keys):
        assert len(vecs)==len(keys), "Dims between vec and key not match"
        return dict([[keys[i],vecs[i]] for i in range(len(vecs))])

    def merge_samples(self,all_feat,feat_sample):
        if(feat_sample.__class__ == dict):
            feat_sample = [feat_sample]
        assert len(all_feat)==len(feat_sample[0]), "Dims Between all_feat and feat_sample not Match"
        for k in all_feat.keys():
            for s in feat_sample:
                all_feat[k].append(s[k])
        return all_feat

    def process(self,to_file=None):
        feat_fact = Feature_Factory('data/shang_index_extra.cfg')
        #init all_feats
        all_feats = map(lambda x: [x,[]],self.feat_conf)
        all_feats = dict(all_feats)

        if(to_file):
            to_file_path = to_file
            to_file = open(to_file,'a')
        print('\n'.join(self.paths))
        print('%d file path' % len(self.paths))
        
        print('>'*15+'Original Features Calculation'+'<'*15)
        print('Number of feat_conf is %d' % len(self.feat_conf))
        print(all_feats)
        i = 0
        for path in self.paths:
            for line in open(path):
                line = line.strip().split('\1')
                feats_vec = line[:]
                assert len(feats_vec)==len(all_feats), "Dims Between Feat and Conf not Match"
                dic_feat = self.map_feat(feats_vec,self.feat_conf)
                #print('Number of feats is %d' % len(feats_vec))
                all_feats = self.merge_samples(all_feats,dic_feat)
                i += 1
        print("Total %d Sample of Daily Stock Data" % i)
        self.all_feats = all_feats

        #calculate extra features for daily stock data
        print('>'*15+'Extra Features Calculation'+'<'*15)
        print('Number of feat_extra_conf is %d' % len(self.feat_extra_conf))
        print('\n'.join(self.feat_extra_conf),'\n')
        if(not all_feats.has_key('returns')):
            returns = feat_fact.calculate_feat(all_feats,spec='returns')
            all_feats.update(returns)
        extra_feats = feat_fact.calculate_feat(all_feats)
        print('========= All Extra Features Done =========')
        print(extra_feats.keys())
        all_feats.update(extra_feats)
        print(all_feats.keys())
        print(len(all_feats))
        if(to_file):
            print('========= To File %s =========' % to_file_path)
            keys = all_feats.keys()
            vecs = all_feats.values()
            #vecs = np.array(zip(*vecs))
            vecs = np.transpose(vecs)
            #print(' '.join(keys))
            #print(' '.join(map(str,vecs[0])))
            print(vecs.shape)
            vecs = map(lambda x: json.dumps(self.map_feat(x,keys)),vecs)
            map(lambda line: to_file.write(line+'\n'),vecs)
        return 0

if __name__ == '__main__':
    test = All_Features(['data/shang_index.txt'],'data/shang_index.cfg','data/shang_index_extra.cfg')
    test.process('test.out')
