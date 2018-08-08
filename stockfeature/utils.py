#-*-encoding=utf-8-*-

import numpy as np

def print_feat(feat,total_length=30):
    prelen = (total_length-len(feat))/2
    feat = '>'*prelen+' '+feat+' '+'<'*(total_length-len(feat)-prelen)
    print(feat)

def normalize(vec,mode='u'):
    vec = np.array(vec,dtype=float)
    if(mode=='u'):
        return (vec-np.min(vec))/(np.max(vec)-np.min(vec))
    if(mode=='z'):
        return (vec-np.average(vec))/np.std(vec)
    else:
        return None
