#-*-encoding=utf-8-*-

'''
Generate Feature for Stock Prediction Model

'''

import numpy as np
import json

class Feature(object):
    '''
    Class for Generating Features for Stock Prediction Model with config
    '''
    def __init__(self,dpath,feat_config,model_config):
        self.f_conf = open(feat_config)
        self.m_conf = open(feat_config)

        self.dpath  = dpath
        self.f_conf = json.loads(self.f_conf)
        print('feature config: ',self.f_conf)
        self.m_conf = json.loads(self.m_conf)
        print('model config: ',self.m_conf)
        
    def generate_feat(self,to_file=None):
        feats_ = {}
        for path in self.dpath:
            for line in open(path):
                try:
                    line = json.loads(line)
                except:
                    line = line.strip().split('\t')
                    date = line[0]
                    feats = line[1].split() 


                    
                    




