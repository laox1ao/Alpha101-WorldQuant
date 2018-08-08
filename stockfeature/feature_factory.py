#-*-encoding=utf-8-*-

import json
import numpy as np
import talib
from Alpha101 import get_alpha
from utils import *
#from talib.abstract import *

'''
All Features For Stock Prediction
    >Original Features: 收盘价,成交量 etc 
    >TA-Lib Features: MAXX,MACD,KDJ,RSI,BOLL etc
    >World-Quant Features: pass
    >News Features: pass
'''

class Feature_Factory(object):
    '''
    '''
    def __init__(self,conf=None):
        self.feat_conf = None
        if(conf):
            self.feat_conf = open(conf).readline().strip().split()
        
    def calculate_feat(self,all_feat,spec=None):
        res = {}
        if(spec):
            if(spec.lower()=='returns'):
                print_feat(spec.upper())
                close = all_feat['closing_price']
                val = map(lambda x: (float(close[x])-float(close[x-1]))/float(close[x-1]),range(1,len(close)))
                val = [np.nan] + val
                print('Returns dim: ',len(val))
                res['returns'] = val
            return res
        close = np.array(all_feat['closing_price'],dtype=float)
        for feat in self.feat_conf:
            #if(feat!='alpha6'): continue
            if(feat.startswith('MA')):
                if(feat=='MACD'):
                    print_feat(feat)
                    res[feat] = val
                    macd, macdsignal, macdhist = talib.MACD(close)
                    print(macd.shape)
                    print(macdsignal.shape)
                    print(macdhist.shape)
                    res['MACD'] = macd
                    res['MACDSignal'] = macdsignal
                    res['MACDHist'] = macdhist
                else:
                    print_feat(feat)
                    day = int(feat[2:])
                    val = talib.MA(close,timeperiod=day)
                    print(val.shape)
                    res[feat] = val
            elif(feat == 'BOLL'):
                print_feat(feat)
                upper, middle, lower = talib.BBANDS(close)
                print(upper.shape)
                print(middle.shape)
                print(lower.shape)
                res['UPPERBAND'] = upper
                res['MIDDLEBAND'] = middle
                res['LOWERBAND'] = lower
            elif(feat == 'KDJ'):
                print_feat(feat)
                pass
            elif(feat == 'RSI'):
                print_feat(feat)
                val = talib.RSI(close)
                print(val.shape)
                res[feat] = val
            elif(feat.startswith('alpha')):
                fact = int(feat[5:])
                val = get_alpha(all_feat,fact)
                res[feat] = val
        return res
