import numpy as np
import pandas as pd
from utils import *

FEAT_FLAG = 'ALpha Factor '

def a_delay(x,d,nd=False):
    #print("%d days(dims) of x reduction" % d)
    res = x[:-d]
    return res if not nd else np.array(res)
def a_correlation(x,y,d,nd=False):
    #print("%d days(dims) of x reduction" % (d-1))
    res = map(lambda t: np.correlate(x[t:t+d],y[t:t+d]).tolist()[0],range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_covariance(x,y,d,nd=False):
    res = map(lambda t: np.cov(x[t:t+d],y[t:t+d])[0][1],range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_scale(x,a=1,nd=False):
    x = np.array(x)
    res = a*x/np.sum(np.abs(x))
    return res.tolist() if not nd else np.array(res)
def a_delta(x,d,nd=False):
    res = map(lambda t: x[t+d]-x[t],range(len(x)-d))
    return res if not nd else np.array(res)
def a_signedpower(x,a,nd=False):
    res = np.power(x,a).tolist()
    return res if not nd else np.array(res)
def a_decay_linear(x,d,nd=False):
    w = range(1,d+1)
    w.reverse()
    w = np.array(w)/np.sum(w)
    res = map(lambda t: np.sum(np.multiply(np.array(x[t:t+d]),w)),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_ts_min(x,d,nd=False):
    res = map(lambda t: np.min(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_ts_max(x,d,nd=False):
    res = map(lambda t: np.max(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_ts_argmin(x,d,nd=False):
    res = map(lambda t: np.argmin(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_ts_argmax(x,d,nd=False):
    res = map(lambda t: np.argmax(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_ts_rank(x,d,nd=False):
    res = map(lambda t: a_rank(x[t:t+d])[-1],range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_sum(x,d,nd=False):
    res = map(lambda t: np.sum(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_product(x,d,nd=False):
    res = map(lambda t: np.product(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_stddev(x,d,nd=False):
    res = map(lambda t: np.std(x[t:t+d]),range(len(x)-d+1))
    return res if not nd else np.array(res)
def a_rank(x,nd=False):
    res = np.array(pd.DataFrame(x).rank(pct=True)).squeeze().tolist()
    return res if not nd else np.array(res)
def a_adv(x,d,nd=False):
    res = map(lambda t: np.mean(x[t:t+20]),range(len(x)-d+1))
    return res if not nd else np.array(res)

def get_alpha(feat,fact):
    #print('Days of feat: %d' % len(feat))
    feat_ = feat
    close = np.array(feat_['closing_price'],dtype=float)
    returns = np.array(feat_['returns'],dtype=float)
    open_price = np.array(feat_['opening_price'],dtype=float)
    low = np.array(feat_['lowest_price'],dtype=float)
    volume = np.array(feat_['volume'],dtype=float)
    amout = np.array(feat_['amount'],dtype=float)
    high = np.array(feat_['highest_price'],dtype=float)
    size = len(close)
    val = None
    if(fact==1):
        print_feat(FEAT_FLAG+str(fact))
        std = a_stddev(close,20)
        step1 = map(lambda x: std[x] if returns[x+19]<0 else close[x],range(len(std)))
        step2 = np.power(step1,2)
        step3 = a_ts_argmax(step2,5)
        step4 = a_rank(step3,nd=True) - 0.5
        val = np.concatenate([[np.nan]*(size-len(step4)),step4])
        print(len(val))
    elif(fact==2):
        print_feat(FEAT_FLAG+str(fact))
        part1 = np.log(volume)
        part1 = a_delta(part1,2)
        part1 = a_rank(part1)
        part2 = (close-open_price)/open_price
        part2 = a_rank(part2)
        val = a_correlation(part1,part2[2:],6)
        val = map(lambda x: -x,val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==3):
        print_feat(FEAT_FLAG+str(fact))
        part1 = a_rank(open_price)
        part2 = a_rank(volume)
        val = a_correlation(part1,part2,10)
        val = map(lambda x: -x,val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==4):
        print_feat(FEAT_FLAG+str(fact))
        step1 = a_ts_rank(a_rank(low),9)
        val = map(lambda x: -x,step1)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==5):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==6):
        print_feat(FEAT_FLAG+str(fact))
        val = a_correlation(open_price,volume,10)
        val = map(lambda x: -x,val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==7):
        print_feat(FEAT_FLAG+str(fact))
        part1 = a_ts_rank(np.abs(a_delta(close,7)),60)
        part1 = map(lambda x: -x,part1)
        part2 = np.sign(a_delta(close,7))[59:]
        val = part1*part2
        adv20 = a_adv(volume,20)[47:]
        cur_volume = volume[66:]
        val = map(lambda t: val[t] if adv20[t]<volume[t] else -1,range(len(val)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==8):
        print_feat(FEAT_FLAG+str(fact))
        part1 = a_sum(open_price,5,nd=True)*a_sum(returns,5,nd=True)
        part2 = a_delay(part1,10,nd=True)
        val = part1[10:] - part2
        val = a_rank(val)
        val = map(lambda t: -t,val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==9):
        print_feat(FEAT_FLAG+str(fact))
        idelta = a_delta(close,1,nd=True)
        itsmin = a_ts_min(idelta,5)
        itsmax = a_ts_max(idelta,5)
        idelta = idelta[4:]
        val = map(lambda t: idelta[t] if 0<itsmin[t] else idelta[t] if itsmax[t]<0 else -1*idelta,range(len(idelta)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==10):
        print_feat(FEAT_FLAG+str(fact))
        idelta = a_delta(close,1)
        itsmin = a_ts_min(idelta,4)
        itsmax = a_ts_max(idelta,4)
        idelta = idelta[3:]
        val = map(lambda t: idelta[t] if 0<itsmin[t] else idelta[t] if itsmax[t]<0 else -1*idelta[t],range(len(idelta)))
        val = a_rank(val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val)) 
    elif(fact==11):
        print_feat(FEAT_FLAG+str(fact))
        val = None
    elif(fact==12):
        print_feat(FEAT_FLAG+str(fact))
        idelta1 = a_delta(volume,1,nd=True)
        idelta2 = -1*a_delta(close,1,nd=True)
        val = np.sign(idelta1*idelta2).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==13):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(close)
        rank2 = a_rank(volume)
        cov = a_covariance(rank1,rank2,5)
        val = a_rank(cov)
        val = map(lambda x: -x,val)
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==14):
        print_feat(FEAT_FLAG+str(fact))
        idelta = a_delta(returns,3)
        rank1 = -1*a_rank(idelta,nd=True)[6:]
        corre = a_correlation(open_price,volume,10,nd=True)
        val = (rank1*corre).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==15):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(high)
        rank2 = a_rank(volume)
        corre = a_correlation(rank1,rank2,3)
        rank3 = a_rank(corre)
        sum1 = -1*a_sum(rank3,3,nd=True)
        val = sum1.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==16):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(high)
        rank2 = a_rank(volume)
        corre = a_correlation(rank1,rank2,5)
        rank3 = -1*a_rank(corre,nd=True)
        val = rank3.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==17):
        print_feat(FEAT_FLAG+str(fact))
        adv20 = a_adv(volume,20,nd=True)
        rank1 = -1*a_rank(a_ts_rank(close,10),nd=True)[14:]
        rank2 = a_rank(a_delta(a_delta(close,1),1))[21:]
        rank3 = a_rank(a_ts_rank((volume[19:]/adv20),5))
        val = (rank1*rank2*rank3).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==18):
        print_feat(FEAT_FLAG+str(fact))
        stddev1 = a_stddev(np.abs(close-open_price),5,nd=True)[5:]
        devi = close-open_price
        corre1 = a_correlation(close,open_price,10,nd=True)
        rank1 = -1*a_rank(stddev1+devi[9:]+corre1,nd=True)
        val = rank1.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==19):
        print_feat(FEAT_FLAG+str(fact))
        delay1 = a_delay(close,7,nd=True)
        delta1 = a_delta(close,7,nd=True)
        part1 = -1*np.sign(close[7:]-delay1+delta1)[242:]
        rank1 = a_rank(1+a_sum(returns,250,nd=True),nd=True)
        part2 = 1+rank1
        val = (part1*part2).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==20):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = -1*a_rank(open_price[1:]-a_delay(high,1),nd=True)
        rank2 = a_rank(open_price[1:]-a_delay(close,1),nd=True)
        rank3 = a_rank(open_price[1:]-a_delay(low,1),nd=True)
        val = (rank1*rank2*rank3).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==21):
        print_feat(FEAT_FLAG+str(fact))
        adv20 = a_adv(volume,20)
        cond1 = (a_sum(close,8,nd=True)/8+a_stddev(close,8)) < (a_sum(close,2,nd=True)[6:]/2)
        cond1 = cond1[12:]
        cond2 = (a_sum(close,2,nd=True)[6:]/2) < (a_sum(close,8,nd=True)/8 - a_stddev(close,8))
        cond2 = cond2[12:]
        cond3 = 1 <= (volume[19:]/adv20)
        val = map(lambda t: -1 if cond1[t] else 1 if cond2[t] else 1 if cond3[t] else -1,range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==22):
        print_feat(FEAT_FLAG+str(fact))
        delta1 = a_delta(a_correlation(high,volume,5),5)
        rank1 = a_rank(a_stddev(close,20),nd=True)
        val = -1*rank1*delta1[10:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==23):
        print_feat(FEAT_FLAG+str(fact))
        cond1 = a_sum(high,20,nd=True)/20 < high[19:]
        delta1 = -1*a_delta(high,2,nd=True)[17:]
        val = map(lambda t: delta1[t] if cond1[t] else 0,range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==24):
        print_feat(FEAT_FLAG+str(fact))
        cond1 = a_delta(a_sum(close,100,nd=True)/100.0,100,nd=True)
        cond1 = cond1 / a_delay(close,100)[99:]
        cond1 = cond1 <= 0.05
        res1 = -1*(close[99:]-a_ts_min(close,100,nd=True))[100:]
        res2 = -1*a_delta(close,3,nd=True)[196:]
        val = map(lambda x: res1[x] if cond1[x] else res2[x],range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==25):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==26):
        print_feat(FEAT_FLAG+str(fact))
        tsrank1 = a_ts_rank(volume,5)
        tsrank2 = a_ts_rank(high,5)
        corre1 = a_correlation(tsrank1,tsrank2,5)
        val = (-1*a_ts_max(corre1,3,nd=True)).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==27):
        print_feat(FEAT_FLAG+str(fact))
        #cond1 = a_rank(a_sum(a_correlation(a_rank(volume),a_rank(vwap))))
    elif(fact==28):
        print_feat(FEAT_FLAG+str(fact))
        adv20 = a_adv(volume,20)
        corre1 = a_correlation(adv20,low[19:],5,nd=True)
        val = a_scale(corre1 - (high+low)[23:]/2 - close[23:])
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==29):
        print_feat(FEAT_FLAG+str(fact))
        part1 = a_ts_min(a_product(a_rank(a_rank(a_scale(np.log(a_sum(a_ts_min(a_rank(a_rank(-1*a_rank(a_delta(close-1,5),nd=True))),2),1))))),1),5)
        part2 = a_ts_rank(a_delay(-1*returns,6),5,nd=True)
        val = (part1 + part2).tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==30):
        print_feat(FEAT_FLAG+str(fact))
        sign1 = np.sign(close[1:]-a_delay(close,1))
        sign2 = np.sign(a_delay(close,1)[1:]-a_delay(close,2))
        sign3 = np.sign(a_delay(close,2)[1:]-a_delay(close,3))
        rank1 = a_rank(sign1[2:]+sign2[1:]+sign3,nd=True)
        val = (1.0-rank1)[1:]*a_sum(volume,5)
        val = val[15:] / a_sum(volume,20)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==31):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(a_rank(a_rank(a_decay_linear(-1*a_rank(a_rank(a_delta(close,10)),nd=True),10))))
        rank2 = a_rank(-1*a_delta(close,3,nd=True),nd=True)
        sign1 = np.sign(a_scale(a_correlation(a_adv(volume,20),low[19:],12)))
        val = rank1[11:] + rank2[27:] + sign1
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==32):
        print_feat(FEAT_FLAG+str(fact))
        #scale1 = a_scale(a_sum(close,7)/7-close[6:])
        #scale2 = 20*a_scale(a_correlation(vwap,a_delay(close,5),230))
        #val = scale1 + scale2
    elif(fact==33):
        print_feat(FEAT_FLAG+str(fact))
        val = a_rank(-1*(1-(open_price/close)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==34):
        print_feat(FEAT_FLAG+str(fact))
        part1 = 1-a_rank(a_stddev(returns,2,nd=True)[3:]/a_stddev(returns,5,nd=True),nd=True)
        part2 = 1-a_rank(a_delta(close,1),nd=True)
        val = a_rank(part1+part2[3:])
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==35):
        print_feat(FEAT_FLAG+str(fact))
        part1 = a_ts_rank(volume,32,nd=True)*(1-a_ts_rank(close+high-low,16,nd=True))[16:]
        part2 = 1-a_ts_rank(returns,32,nd=True)
        val = part1*part2
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==36):
        print_feat(FEAT_FLAG+str(fact))
        #rank1 = 2.21*a_rank(a_correlation(close-open_price,a_delay(volume,1),15),nd=True)
        #rank2 = 0.7*a_rank(open_price-close,nd=True)
        #rank3 = 0.73*a_rank(a_ts_rank(delay(-1*returns,6),5),nd=True)
        #rank4 = a_rank(np.abs(a_correlation(vmap,a_adv(volume,20),6)),nd=True)
        #rank5 = 0.6*a_rank((a_sum(close,200,nd=True)/200-open_price)*(close-open_price),nd=True)
        #val = rank1 + rank2 + rank3 + rank4 + rank5
    elif(fact==37):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(a_correlation(a_delay(open_price-close,1),close[1:],200),nd=True)
        rank2 = a_rank(open_price-close,nd=True)
        val = rank1 + rank2[200:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==38):
        print_feat(FEAT_FLAG+str(fact))
        val = -1*a_rank(a_ts_rank(close,10),nd=True)*a_rank(close/open_price)[9:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==39):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = -1*a_rank(a_delta(close,7)[20:]*(1-a_rank(a_decay_linear(volume[19:]/a_adv(volume,20),9),nd=True)),nd=True)
        rank2 = 1+a_rank(a_sum(returns,250),nd=True)
        val = rank1[222:]*rank2
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==40):
        print_feat(FEAT_FLAG+str(fact))
        val = -1*a_rank(a_stddev(high,10),nd=True)*a_correlation(high,volume,10)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==41):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==42):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==43):
        print_feat(FEAT_FLAG+str(fact))
        tsrank1 = a_ts_rank(volume[19:]/a_adv(volume,20),20,nd=True)
        tsrank2 = a_ts_rank(-1*a_delta(close,7,nd=True),8,nd=True)
        val = tsrank1*tsrank2[24:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==44):
        print_feat(FEAT_FLAG+str(fact))
        val = -1*a_correlation(high,a_rank(volume),5,nd=True)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==45):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(a_sum(a_delay(close,5),20,nd=True)/20)
        corre1 = a_correlation(close,volume,2,nd=True)
        rank2 = a_rank(a_correlation(a_sum(close,5)[15:],a_sum(close,20),2))
        val = rank1*corre1[23:]*rank2[4:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==46):
        print_feat(FEAT_FLAG+str(fact))
        core = (a_delay(close,20,nd=True)-a_delay(close,10,nd=True)[10:])/10 - (a_delay(close,10,nd=True)-close[10:])[10:]/10
        cond1 = core > 0.25
        cond2 = core < 0
        val = map(lambda t: -1 if cond1[t] else 1 if cond2[t] else -1*(close[1:]-a_delay(close,1))[19:],range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==47):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==48):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==49):
        print_feat(FEAT_FLAG+str(fact))
        core = (a_delay(close,20,nd=True)-a_delay(close,10,nd=True)[10:])/10 - (a_delay(close,10,nd=True)-close[10:])[10:]/10
        cond1 = core < -0.1
        val = map(lambda t: 1 if cond1[t] else -1*(close[1:]-a_delay(close,1)),range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==50):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==51):
        print_feat(FEAT_FLAG+str(fact))
        core = (a_delay(close,20,nd=True)-a_delay(close,10,nd=True)[10:])/10 - (a_delay(close,10,nd=True)-close[10:])[10:]/10
        cond1 = core < -0.05
        val = map(lambda t: 1 if cond1[t] else -1*(close[1:]-a_delay(close,1)),range(len(cond1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==52):
        print_feat(FEAT_FLAG+str(fact))
        part1 = -1*a_ts_min(low,5,nd=True)[5:] + a_delay(a_ts_min(low,5),5)
        rank1 = a_rank((a_sum(returns,240,nd=True)-a_sum(returns,20)[220:])/220)
        tsrank1 = a_ts_rank(volume,5)
        val = part1[230:]*rank1*tsrank1[235:]
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==53):
        print_feat(FEAT_FLAG+str(fact))
        val = -1*a_delta((close-low-high+close)/(close-low),9,nd=True)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==54):
        print_feat(FEAT_FLAG+str(fact))
        val = -1*((low-close)*np.power(open_price,5))/((low-high)*np.power(close,5))
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==55):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank((close[11:]-a_ts_min(low,12))/(a_ts_max(high,12,nd=True)-a_ts_min(low,12)))
        rank2 = a_rank(volume)
        val = -1*a_correlation(rank1,rank2[11:],6,nd=True)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==56):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==57):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==58):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==59):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==60):
        print_feat(FEAT_FLAG+str(fact))
        scale1 = 2*a_scale(a_rank(((close-low-high+close)/(high-low))*volume),nd=True)
        scale2 = a_scale(a_rank(a_ts_argmax(close,10)),nd=True)
        val = -1*(scale1[9:]-scale2)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==61):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==62):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==63):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==64):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==65):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==66):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==67):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==68):
        print_feat(FEAT_FLAG+str(fact))
        tsrank1 = a_ts_rank(a_correlation(a_rank(high)[14:],a_rank(a_adv(volume,15)),8),13,nd=True)
        rank1 = a_rank(a_delta(close*0.518371+low*(1-0.518371),1),nd=True)
        val = -1*(tsrank1 < rank1[32:])
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==69):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==70):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==71):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==72):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==73):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==74):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==75):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==76):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==77):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==78):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==79):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==80):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==81):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==82):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==83):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==84):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==85):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(a_correlation((high*0.876703+close*(1-0.876703))[29:],a_adv(volume,30),9),nd=True)
        rank2 = a_rank(a_correlation(a_ts_rank((high+low)/2,3)[7:],a_ts_rank(volume,10),7),nd=True)
        val = np.power(rank1,rank2[22:])
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==86):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==87):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==88):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(a_decay_linear(a_rank(open_price,nd=True)+a_rank(low)-a_rank(high)-a_rank(close),8),nd=True)[84:]
        tsrank1 = a_ts_rank(a_decay_linear(a_correlation(a_ts_rank(close,8)[71:],a_ts_rank(a_adv(volume,60),20),8),6),2,nd=True)
        val = map(lambda t: rank1[t] if rank1[t]<tsrank1[t] else tsrank1[t],range(len(rank1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==89):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==90):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==91):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==92):
        print_feat(FEAT_FLAG+str(fact))
        tsrank1 = a_ts_rank(a_decay_linear(((high+low)/2+close)<(low+open_price),14),18)[15:]
        tsrank2 = a_ts_rank(a_decay_linear(a_correlation(a_rank(low)[29:],a_rank(a_adv(volume,30)),7),6),6)
        val = map(lambda t: tsrank1[t] if tsrank1[t] < tsrank2[t] else tsrank2[t],range(len(tsrank1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==93):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==94):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==95):
        print_feat(FEAT_FLAG+str(fact))
        rank1 = a_rank(open_price[11:]-a_ts_min(open_price,12),nd=True)
        sum1 = a_sum((high+low)/2,19)[39:]
        sum2 = a_sum(a_adv(volume,40),19)
        tsrank1 = a_ts_rank(np.power(a_rank(a_correlation(sum1,sum2,12)),5),11)
        val = rank1[67:] < tsrank1
        val = val.tolist() 
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==96):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==97):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==98):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==99):
        print_feat(FEAT_FLAG+str(fact))
        sum1 = a_sum((high+low)/2,19)[59:]
        sum2 = a_sum(a_adv(volume,60),19)
        rank1 = a_rank(a_correlation(sum1,sum2,8))
        rank2 = a_rank(a_correlation(low,volume,6))
        val = map(lambda t: rank1[t] if rank1[t] < rank2[t] else rank2[t],range(len(rank1)))
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    elif(fact==100):
        print_feat(FEAT_FLAG+str(fact))
    elif(fact==101):
        print_feat(FEAT_FLAG+str(fact))
        val = (close-open_price)/(high-low+ 0.001)
        val = val.tolist()
        val = [np.nan]*(size-len(val)) + val
        print(len(val))
    else:
        pass
        #print('Not Understood factor, accept 1~101')
    return val

if __name__ == '__main__':
    num = 300
    feat = {
            "closing_price": np.ones([num]),
            "returns": np.random.randn(num),
            "opening_price": np.ones([num]),
            "volume": np.ones([num])*100,
            "amount": np.ones([num])*110,
            "lowest_price": np.ones([num])*0.5,
            "highest_price": np.ones([num])*1.5,
            }
    for i in range(1,102):
        get_alpha(feat,i)
    #print(a_ts_rank(np.array(range(10)),3))
    #print(a_rank(np.array(range(10))))
