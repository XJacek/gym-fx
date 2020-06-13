
import pandas as pd
import numpy as np
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
import ta


#from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
 
scaler = MinMaxScaler(feature_range=(-1, 1), copy=True) 
#scaler = preprocessing.RobustScaler(copy=True, quantile_range=(0.0, 100.0), with_centering=True,with_scaling=True)
#scaler = MaxAbsScaler()
#scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
#scaler = preprocessing.QuantileTransformer(output_distribution='uniform') #

def get_ohlc_features( ohlcdata):
    np.set_printoptions(precision=6) 
    #print('ohlcdata',ohlcdata)
    ohlc=ohlcdata.dropna().reset_index(drop=True).values # to numpy
         
    Timestamp = 0#ohlc[:, 1]
    Open = ohlc[:, 2] #.astype(float)#2
    High = ohlc[:, 3] #.astype(float)#2
    Low = ohlc[:, 4] #.astype(float)#2
    Close = ohlc[:, 5] #.astype(float)#2
    #print('close',Close)
    res = np.zeros(shape=(len(range(0, len(Open), 1)), 5))
 
    for it in range(len(ohlc)):
        res[it,0] = 0#float(Timestamp[it].strftime("%H"))  # time Hour
        res[it,1] = Open[it]  # open
        res[it,2] = Low[it]  # close
        res[it,3] = High[it]  # low
        res[it,4] = Close[it]  # high
    #print ('res',res)
    return res       


    
def get_tickbars_features( ticks, frequency, diff_ticks=False, residual_filter=False):

    ticks=ticks.dropna().reset_index(drop=True).values
         
    times = ticks[:, 0] #
    pricesBid = ticks[:, 1].astype(float)#
    pricesAsk = ticks[:, 2].astype(float) #
    res = np.zeros(shape=(len(range(0, len(pricesBid), frequency)), 5))
    it = 0
    i=0
               
    while i<=len(pricesBid) :
        if ( i % frequency == 0 or i == len(pricesBid) ) and i!=0 :
            if i == len(pricesBid) : frequency=i
            windowA = pricesAsk[i - frequency:i]
            windowB = pricesBid[i - frequency:i]            
            
            if diff_ticks:    
                windowA = np.diff(pricesAsk[i - frequency:i])
                windowB = np.diff(pricesBid[i - frequency:i]) 
            if residual_filter:
                windowRA = residual_analysis(windowA, freq=5, show=False)
                try: 
                    if windowRA==0: windowA=windowA
                except: windowA=windowRA 
                windowRB = residual_analysis(windowB, freq=5, show=False)
                try: 
                    if windowRB==0: windowB=windowB
                except: windowB=windowRB
            try:
                maxp = np.max(windowA) # High
                maxarg=np.where( windowA == maxp )[0][0]
            except:maxarg=0
            try:
                minp = np.min(windowA) # Low
                minarg=np.where( windowA == minp )[0][0]
            except:minarg=0

            if maxarg<minarg: HL_seq = 1  #  Highest first
            elif maxarg>minarg: HL_seq = -1 #  Lowest first
            else: HL_seq = 0
            time = np.arange(frequency)
            data = np.array([windowA, time, windowB])

            res[it][0] = 0#float(times[i-1].strftime("%H"))  # time Hour
            res[it][1] = windowA[0]  # open
            res[it][2] = windowA[-1:]  # close
            res[it][3] =minp  # low
            res[it][4] =maxp  # high
            it += 1
        i+=1
    return res


def get_candles_features(res, frac_diff=False, fast_frac_diff =False, d=0.4, thres=1e-5, lim=1e5,  log = False, diff=False):
    
    #convert np to pd
    dfi = pd.DataFrame({'Open': res[:, 1],'High': res[:, 2],'Low': res[:, 3],'Close': res[:, 4]})      
    dfk=dfi.copy()
    #print('91834 dfi',dfi.shape)
    #dfi['ta1'] =  ema_indicator(dfi["Close"], n=10, fillna=True).astype(float)
    dfi['ta1'] =  kama(dfi["Close"], n=5, fillna=True).astype(float)
    dfi['ta2'] =  kama(dfi["Close"], n=50, fillna=True).astype(float)
    dfi['ta3'] =  kama(dfi["Close"], n=100, fillna=True).astype(float)
    #dfi['ta1'] =  bollinger_mavg(dfi["Close"], n=10, fillna=True).astype(float)
    #dfi['ta2'] =  bollinger_hband(dfi["Close"], n=10, ndev=2,fillna=True).astype(float)
    #dfi['ta3'] =  bollinger_lband(dfi["Close"], n=10, ndev=2,fillna=True).astype(float)
    # log
    if log: dfi=np.log(dfi)
    #diff
    if diff: dfi=dfi.diff()
    else:dfi=dfi
    

    if frac_diff: dfi = fracDiff_FFD(dfi ,d=d,thres=thres) 
    
    if fast_frac_diff: dfi = fast_fracDiff_FFD(dfi ,d=d) 

   
    dfi['ta4'] = rsi(dfk["Close"], n=5,fillna=True).astype(float)
    dfi['ta5'] = rsi(dfk["Close"], n=15,fillna=True).astype(float)
    dfi['ta6'] = stoch_signal(dfk["Low"],dfk["High"],dfk["Close"], n=5,fillna=True).astype(float)
    dfi['ta7'] = stoch_signal(dfk["Low"],dfk["High"],dfk["Close"], n=15,fillna=True).astype(float)
    dfi['ta8'] = macd(dfk["Close"], n_fast=5, n_slow=15, fillna=True) # adx(dfi["High"],dfi["Low"],dfi["Close"], n=10,fillna=True).astype(float)

    #print(dfi)
    if frac_diff: 
        width = get_width(d,thres)

    elif fast_frac_diff:
        width = fast_get_width(d,thres,lim)
        dfi = dfi.iloc[width:,]
    else: width = 0
    data = res[width:,] #  #from fracDiff_FFD
    df = pd.DataFrame(
        { 'Open': dfi["Open"],'High': dfi["High"], 'Low': dfi["Low"], 'Close': dfi["Close"],
            'ta1': dfi['ta1'],'ta2': dfi['ta2'],'ta3': dfi['ta3'],'ta4': dfi['ta4'],'ta5': dfi['ta5'],
            'ta6': dfi['ta6'],'ta7': dfi['ta7'],'ta8': dfi['ta8']
        }) 

    df = pd.DataFrame(df, columns= df.columns).reset_index(drop=True)
    df = df.replace([np.inf], 99)
    df = df.replace([-np.inf], -99)
    df = df.fillna(0)
    df_scaled = pd.DataFrame(scaler.fit_transform(df),columns = df.columns)
    #print('102883', df_scaled.shape)
    return df_scaled


def generate_obs(active_df):

    obs = np.array([
        
        active_df['Open'].values.astype(float),     #0
        active_df['High'].values.astype(float),        
        active_df['Low'].values.astype(float),
        active_df['Close'].values.astype(float),        
        active_df['ta4'].values.astype(float),
        active_df['ta5'].values.astype(float),
        active_df['ta6'].values.astype(float),   #35
        active_df['ta7'].values.astype(float),
        active_df['ta8'].values.astype(float),
        #active_df['Timestamp'].values.astype(float), #40

        ])

    return obs




         
def getWeights_FFD(d,thres):
    w,k=[1.],1
    while True:
        w_=-w[-1]/k*(d-k+1)
        if abs(w_)<thres:break
        w.append(w_);k+=1
    return np.array(w[::-1]).reshape(-1,1)

#---------------------------------------------------------------------------

# As described in Advances of Machine Learning by Marcos Prado

def fracDiff_FFD(series,d=0.4,thres=1e-5):
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df={}
    for name in series.columns:

        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            test_val = series.loc[loc1,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any(): continue # exclude NAs
            #print(f'd: {d}, iloc1:{iloc1} shapes: w:{w.T.shape}, series: {seriesF.loc[loc0:loc1].notnull().shape}')
            try:
                df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
        
    df=pd.concat(df,axis=1)
    
    return df



def get_width(d,thres):
    w = getWeights_FFD(d,thres)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    return width



# forked from https://github.com/philipperemy/fractional-differentiation-time-series
# from: http://www.mirzatrokic.ca/FILES/codes/fracdiff.py
# small modification: wrapped 2**np.ceil(...) around int()
# https://github.com/SimonOuellette35/FractionalDiff/blob/master/question2.py
def fast_fracdiff(x, d=0.4):
    import pylab as pl
    T = len(x)

    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T])

def get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def fast_get_width(d,thres,lim):
    w = get_weight_ffd(d, thres, lim)    
    width = len(w)-1
    return width



def fast_fracDiff_FFD(series,d):
    df=pd.DataFrame() #{}
    for name in series.columns:

        seriesF=series[name].fillna(method='ffill').dropna().values.T.astype(float)
        fracs = fast_fracdiff(seriesF,d=d)
        df_ = pd.DataFrame(data=fracs.T,columns=[name])   
        df = pd.concat([df,df_],axis=1)     
    return df

