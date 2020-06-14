
import time
import math
from array import *
import gym
import pandas as pd
import numpy as np
from gym import spaces
from collections import OrderedDict
from gym import GoalEnv, spaces
from gym.utils import seeding
#import threading
import threading
#import multiprocessing
import queue

from typing import Any

from gym_fx.utils.MT4Py import ClientMT4

from gym_fx.utils.Features import * #Features as fs

import matplotlib
matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
import matplotlib.pyplot as plt
plt.style.use('bmh')

#from numba import jit

#TODO: Zapisywanie tickow do pliku, ograniczanie rozmiaru tablicy z danymi w thred mt4, czytanie swieczek z mt4


class FxEnv(gym.GoalEnv):
    """ Forex with mt4 support trading enviroment for OpenAI gym"""
    metadata = {'render.modes': ['human', 'system', 'none']}
    
    def __init__(self, df,act_space='discret', obs_space ='box', 
                 lookback=12,n_max_indicators =100,ticks=False, freqency_tickbar = 1000, step=1000, 
                 initial_balance=10000, units=10000, leverage = 500, commission=5, spread=1, stop_out_level = 50, hedge =False,
                 serial=True,  
                 test = False,
                 report_interval = 200,
                 report_in_learn = True,
                 simple_report_in_learn=False,
                 mt4 = False, mt4_symbol = 'EURUSD', magic ='10000', mt4_timeframe=1,
                 diff_ticks=False,
                 frac_diff = False , d =3.4, thres=1e-4,
                 fast_frac_diff = False,                 
                 log = True,
                 diff=True,                 
                 desired_goal_diff=0,
                 warning = False,
                 lot_sizing = False, # if True increase nA = 4*10
                 episod_steps = 60,
                 nA =4 # number of actions: 0-buy 1-sell 2-hold  3 -closeAll 5-closeAllLong, 6-closeAllShort
                 ):
        super(FxEnv, self).__init__()
        
        self.line1=[]
        self.ticks =  ticks
        self.Nlot = units
        self.point = 0.00001        
        self.SL = 50000 * self.point
        self.TP = 50000 * self.point  
        self.simple_report_in_learn = simple_report_in_learn
        if not warning:
            import warnings
            warnings.filterwarnings("ignore")
        #print('max_steps in loaded data = ',len(df)/freqency_tickbar)
        self.Max_allowed = 20 # maximum allowed open trades 
        self.ACTION_SPACE_MULTI = False  # Actions with lots sizing
        if act_space=='discret'  :self.ACTION_SPACE_BOX = False # true = discrete
        elif act_space=='box' : self.ACTION_SPACE_BOX = True # true = discrete
        else:self.ACTION_SPACE_BOX = False

        if obs_space == 'box':self.OBS_SPACE_GOALENV = False # standard obs space
        elif obs_space == 'goalenv':self.OBS_SPACE_GOALENV = True # observation space with GoalEnv Dict for HER algoritm
        else:self.OBS_SPACE_GOALENV = False # 


        self.nA = nA # number of actions buy sell, hold  closeL, closeS
        self.lot_sizing = lot_sizing
        self.episod_steps = episod_steps
        self.step_count = 0
        

        self.spread = spread * self.point
        self.Quote_Home_Currency =  1 # Quote/Home Currency = USD/USD # Currency Pair: GBPUSD, Home Currency: USD
        self.commission =  (self.Nlot * commission)/ 100000 * self.Quote_Home_Currency
        self.leverage = leverage 
        self.stop_out_level = stop_out_level
        self.hedge = hedge
        
        self.lookback = lookback
        self.freqency_tickbar = freqency_tickbar
        self._step = step
        self.n_max_indicators = n_max_indicators  #  adding bars to calculate indicators , highest value from used indicators i.e EMA 20 needs 20 bars

        self.diff_ticks = diff_ticks        
        self.log = log
        self.diff = diff 
        self.diffI = int( self.diff== True  )
        self.frac_diff = frac_diff 
        self.fast_frac_diff = fast_frac_diff
        self.d = d
        self.thres=thres
        if frac_diff:self.frac_width = get_width(self.d,self.thres)+self.diffI
        elif fast_frac_diff:
            lim = 3000 
            self.frac_width = fast_get_width(self.d,self.thres,lim)+self.diffI
        else: self.frac_width = 0


        self.frame_start = self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  # OHLC
        if self.ticks:
            self.frame_start = (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI)* self.freqency_tickbar 
        print("frame start",self.frame_start)







      
        self.mt4= mt4
        self.mt4_symbol =mt4_symbol
        self.mt4_timeframe = mt4_timeframe
        self.magic = magic

        self._delay = 0.1    # repeat execution delay
        self._wbreak = 30    # execution timeout
        self.interval = 0.003 # interval data read by thread
        self._verbose = False

        self.report_interval = report_interval
        self.report_in_learn = report_in_learn
        
        self.observation=np.full((1,4),0)

        if not self.ticks and self.mt4==False:
           # self.dfticks = df.dropna().reset_index().values
            #self.dfticks  = self.dfticks.values
            self.df = df # .dropna().reset_index().values
            
        if self.ticks and self.mt4==False:
           # self.dfticks = df.dropna().reset_index().values
            #self.dfticks  = self.dfticks.values
            self.df = df # .dropna().reset_index().values
            self.df[['Data','Timestamp']] = self.df.Timestamp.str.split(" ",expand=True)

            #print('ddddddddddddddddddddd')
            print(self.df)

        if self.mt4:
            self._sMt4 = ClientMT4(verbose=True)
            #self._sMt4 = ClientMT4(verbose=True)
            #self. runMT4(self.frame_start) ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.ThreadingMT4(self.interval)
            if self.ticks  :            
            
            
                self.df_raw = pd.DataFrame()

                rows = (self.lookback  + self.n_max_indicators  + self.frac_width )* self.freqency_tickbar
                dfp = self.df_raw.copy()
                lenprev = 0
                while len(dfp.index) <= rows:
                
                    self.inter_step = len(dfp.index)
                    try:
                        self.current_Ask = dfp.iloc[-1,1]
                        self.current_Bid = dfp.iloc[-1,2]                
                    
                    except: self.current_Ask=self.current_Bid=0
                    if self.inter_step!= lenprev: print('collecting ticks', self.inter_step,'/',rows,
                           '  ',self.mt4_symbol +' '+ str(self.current_Ask) +'/'+ str(self.current_Bid) )
                    lenprev = self.inter_step
                    dfp = self.df_raw.copy()             
                    time.sleep(self.interval)
                   # print('df_lastt.shape', df_last.shape)

                print('df.shape', dfp.shape)
                dfp.columns = ['Timestamp', 'Bid price', 'Ask price']
                dfp['Timestamp'] =  dfp['Timestamp'].str.replace(':', '')
                dfp['Timestamp'] =  dfp['Timestamp'].str.replace('.', '')
                dfp['Timestamp'] = dfp['Timestamp'].str.replace('-', '')
                dfp['Timestamp'] =  dfp['Timestamp'].str.replace(' ', '')            
                self.df = dfp
                #print('DFFFFFFFFFFFFFFFFFFF',self.df)
                #del dfp
            else:   #OHLC
                while  self._sMt4._DB_sMt4.empty:
                    time.sleep(1)
                self.df = self._sMt4._DB_sMt4.copy()
                #while self.df.empty:
                #    print("Loading OHLC data")
                #    time.sleep(1)
                print("MT4 OHLC",self.df)
                








        self.initial_balance_base = initial_balance  # base currency EUR
        self.balance = self.initial_balance_base
        self.prev_balance = self.prev_balance1 = self.initial_balance_base
        self.balancex100 = 10 * self.initial_balance_base
        self.equity = self.initial_balance_base
        self.prev_equity = self.initial_balance_base
        self.max_equity = self.initial_balance_base
        self.min_equity = self.initial_balance_base
        self.max_balance = self.initial_balance_base
        self.min_balance = self.initial_balance_base
  
        self.prev_max_equity = self.initial_balance_base
        self.prev_min_equity = self.initial_balance_base
        self.prev_max_balance = self.initial_balance_base
        self.prev_min_balance = self.initial_balance_base
        self.freemargin = self.initial_balance_base
        self.floating_profits = 0    
        self.exposure = 0        
        self.longs = 0
        self.shorts = 0
        self.action_typ =-1
        self.exp_counter=-1
        self.loss =200
        self.reward_sum =0
        self.prev_reward_sum =0
        self.equity_curve = np.full((1,4),0) # for visual raport
        self.reward_curve = np.full((1,7*10+1),0)

        self.serial = serial
        
        self.bar_step = 0
        self.tick_step = 0
        self.ohlc_step=0
        if self.ACTION_SPACE_MULTI:

            self.action_space = spaces.MultiDiscrete([self.nA, 10 ]) # [action ,lots] 
        elif self.ACTION_SPACE_BOX:
            #self.action_space = spaces.Box(low=-self.nA, high=self.nA, shape=(1,), dtype=np.float32)
            #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,1), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
            
        else: self.action_space = spaces.Discrete(self.nA*1*1*1)
       
        obs_dim = 9+4 # 14  !!!!!!!!!!!!!  add dim if  you add next indicators etc    

        if self.OBS_SPACE_GOALENV:        
            self.observation_space = spaces.Dict({
                'observation': spaces.Box(low=-1.0, high=1.0, shape=(obs_dim*lookback,  ), dtype=np.float32),
                'achieved_goal': spaces.Box( 0.0, 1., shape=(1,)),
                'desired_goal': spaces.Box( 0.0, 1., shape=(1,))
            })
        else:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_dim * lookback,  ), dtype=np.float32)

        self.trades_counter = 0
        self.longs_floating = 0
        self.shorts_floating = 0
        self.exposure = 0
        self.prev_exposure =0
        self.test = test
        self.done1=0
        self.initial_state = self.initial_balance_base 
        self.desired_goal_diff = desired_goal_diff
        self.success = 0
        self.dg = np.array([(self.initial_balance_base+desired_goal_diff)/self.balancex100]) #desired goal
        self.achieved_goal=0
        self.desired_goal=0

        self.prev_achieved_goal = 0
        self.minpA =0
        self.minpB =0
        self.maxpA =0
        self.maxpB =0
        self.seq=0
        self.rewi=0
        ##----------------------------------------------
               
            


        #self.reset()

    def wait_for_sMt4(self):
        #print("wait_for_sMt4")
        if self.ticks:
            print(':::::::::::::::::::::WAIT for next candle::::::::::::::::')
        
            #print(self.frame_start + self.bar_step,':::::::::::::::::::::self.frame_start + self.bar_step::::::::::::::::')
        
            dfp = self.df_raw.copy()
            lenprev = 0
            while len(dfp.index) <= (self.frame_start + self.bar_step + self.tick_step):  
                if len(dfp.index)!= lenprev:
                    self.inter_step = len(dfp.index)
                    print('collecting ticks next candle', self.inter_step ,'/ ', (self.frame_start + self.bar_step + self.tick_step), 
                           '  ',self.mt4_symbol +'  '+ str(self.current_Ask) +'/'+ str(self.current_Bid) )  
                    self._make_statement()
                lenprev = self.inter_step
                dfp = self.df_raw.copy() 
                #if lenprev>0:
                    #self.current_Ask = dfp.iloc[-1,1]
                    #self.current_Bid = dfp.iloc[-1,2]
                    #print('ask',self.current_Ask)
                    #self.Spread = (self.current_Ask-self.current_Bid)/self.point
            
                time.sleep(self.interval)            
               # print('df_lastt.shape', df_last.shape)        
            #print('dfp.shape', dfp.shape)
            dfp.columns = ['Timestamp', 'Bid price', 'Ask price']
            dfp['Timestamp'] =  dfp['Timestamp'].str.replace(':', '')
            dfp['Timestamp'] = dfp['Timestamp'].str.replace('-', '')
            dfp['Timestamp'] =  dfp['Timestamp'].str.replace(' ', '')            
            self.df = dfp
            #del dfp
            #return df
        else:
            #time.sleep(5)
            print('Waiting for nex bar...')
            #from sys import stdout
            while self._sMt4._DB_sMt4['Timestamp'].values[-1]==self.df['Timestamp'].values[-1]:
                i=("Waiting for next bar")
                #stdout.write("\r%s" % i)
                #stdout.flush()
                #time.sleep(0.22)
                #stdout.write("\n")
                #print('54218',self.df['Timestamp'].values[-1])
            self.df=self._sMt4._DB_sMt4.copy()
            self.df.index = self.df.index + self.ohlc_step +1
            


    def _make_statement(self):
            
        #print("_make_statement")
        #print('629293',self.df.loc[(self.frame_start + self.ohlc_step)]["Close"].astype(float))
              #loc[str(self.frame_start + self.ohlc_step)])
        self.current_Ask = self._get_current_Ask()
        self.current_Bid = self._get_current_Bid()
        self.Spread = (self.current_Ask-self.current_Bid)/self.point

        #self.high_price = self._get_current_high_price()
        #self.low_price = self._get_current_low_price()

        self.prev_Bid = self.current_Bid

         # Calc equity
        floating_long_profit = 0
        floating_short_profit = 0
        self.used_margin = 0

        #Home Currency: USD
        #Currency Pair: GBP/CHF
        #Base = GBP; Quote = CHF
        #Quote / Home Currency = CHF/USD = 1.1025
        #Opening Rate = 2.1443
        #Closing Rate = 2.1452
        #Units = 1000

        #Then:

        #Profit = (2.1452 - 2.1443) * (1.1025) * 1000
        #Profit = 0.99225 USD
        # https://www1.oanda.com/forex-trading/analysis/profit-calculator/how
        
        #https://www.metatrader5.com/en/terminal/help/trading_advanced/margin_forex

        for x in range(0, self.Max_allowed):

            lots1 = self.account_longs_lots[x, -1]
            #print('lots1=',x,lots1)
            Orderprice1 = self.account_longs[x, -1]
            if Orderprice1!=0:

                pricechange1 = (self.current_Bid) - Orderprice1
                profit1 = pricechange1 * lots1 * self.Nlot - self.commission * lots1 
                profit1 = round(profit1, 2)
                floating_long_profit += profit1

                #print('floating_long_profit=',x, floating_long_profit)
                self.used_margin += lots1 * self.req_margin

            lots2 = self.account_shorts_lots[x, -1]
            #print('lots2=',x, lots2)
            Orderprice2 = self.account_shorts[x, -1]
            if Orderprice2!=0:

                pricechange2 =  Orderprice2 - (self.current_Ask )

                profit2 = pricechange2  * lots2 * self.Nlot - self.commission * lots2 
                profit2 = round(profit2, 2)
                floating_short_profit += profit2
                #print('floating_short_profit=',x, floating_short_profit)
                self.used_margin -= lots2 * self.req_margin
                #print(self.account_longs[:, -1])
        
        self.used_margin = abs(self.used_margin)    
        self.floating_profits = floating_long_profit + floating_short_profit  # (self.balance_base + self.balance_quote/current_price)
        self.floating_profits = round(self.floating_profits, 2)

                #Equity = Account Balance + Floating Profits (or Losses)
        self.prev_equity = self.equity
        #self.prev_balance = self.balance

        self.equity =  self.balance + self.floating_profits #
        self.equity= round(self.equity, 2)
        self.freemargin = self.equity - self.used_margin
        #Margin level = (equity/used margin) x 100
        
        
        if self.used_margin!=0:
            self.margin_level = self.equity/self.used_margin * 100
        else: self.margin_level=np.inf

        if self.equity<=0: # Close all trades
            self.close_all_long()                
            self.close_all_short()
            self.equity=self.balance

        self.prev_max_equity = self.max_equity
        self.prev_min_equity = self.min_equity

        if self.equity >= self.max_equity:
            self.max_equity = self.equity
        if self.equity <= self.min_equity:
            #self.max_equity = self.min_equity
            self.min_equity = self.equity

        self.prev_max_balance = self.max_balance
        self.prev_min_balance = self.min_balance

        if self.balance >= self.max_balance:
            self.max_balance = self.balance
        if self.balance <= self.min_balance:
            #self.max_balance = self.min_balance
            self.min_balance = self.balance 
            
        if self.simple_report_in_learn: 
           # print('Step','%.0f' % (self.bar_step/self._step),' Step',self.frame_start+self.bar_step,'  Price','%.5f' %self.current_Bid,'Profit = ', '%.2f' % self.floating_profits,'Equity = ',
            print('Step','%.0f' % (self.step_count),' OHLC step',self.frame_start+self.ohlc_step,'  Price Bid','%.5f' %self.current_Bid,'Profit = ', '%.2f' % self.floating_profits,'Equity = ',
            '%.2f' % self.equity,'Balance = ','%.2f' %  self.balance,'\n','Margin = ', self.used_margin,'FreeMargin = ','%.2f' %  self.freemargin, 'Margin level = ','%.2f' %  self.margin_level+'%',  
            'Spread','%.0f' % self.Spread, ' L/S ', str(self.longs) +' vs '+ str(self.shorts),'\n' ,
            'Trades',self.trades_counter, 'Longs',self.longs_floating,'Shorts', self.shorts_floating,'Exposure',self.exposure, 'successed goals:',self.success)
        
        if self.report_in_learn and self.step_count % self.report_interval==0: self._plot_report()



    def _plot_report11(self):
    
        import matplotlib
        matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        x= len(self.equity_curve) -1
        plt.ion()

 
        
        x = np.arange(self.step_count)
        if self.step_count==1:
            self.fig = plt.figure(1) 
            self.ax5=self.fig.add_subplot(413)
            self.lax5, = self.ax5.plot([],[]) # [:,1:4]
            plt.title("Equity/balance", fontsize=6)
            #plt.show()
        else:
            x =  range(len(self.equity_curve)-1)
            y=self.equity_curve[-1:,0]
            #y1=self.equity_curve[1:,1]
            self.lax5.set_ydata(y)
            #self.lax5.set_ydata(y1)
            self.lax5.set_xdata(x)
            if len(y)>0:
                self.ax5.set_ylim(min(y),max(y)+1) # +1 to avoid singular transformation warning
                self.ax5.set_xlim(min(x),max(x)+1)

            plt.title("Equity/balance", fontsize=6)
            plt.draw()
            plt.pause(0.1)
        #plt.show(block=False)

        #sns.lineplot(data=pd.DataFrame(self.equity_curve[1:,0:2]), x='0',y='1')

  
        #plt.savefig("report"+".png")

    def _plot_report_222(self):
        x= len(self.equity_curve) -1
        plt.ion()
        if self.step_count==1:

            
           

            self.fig = plt.figure(1) #fig_1 = plt.figure(1)
            self.ax1=self.fig.add_subplot(4,3,1)
            plt.title("OHCL_raw", fontsize=6)  
            self.ax2=self.fig.add_subplot(432) 
            plt.title("OHCL scaled as inputs", fontsize=6)
            self.ax3=self.fig.add_subplot(433)
            plt.title("All features", fontsize=6)
            self.ax4=self.fig.add_subplot(412)
            plt.title("Exposure", fontsize=6)
            self.ax5=self.fig.add_subplot(413)
            plt.title("Equity/balance", fontsize=6)
            self.ax6=self.fig.add_subplot(414) 
            plt.title("Price Chart", fontsize=6)
            self.ax6.grid()
            self.ax7=self.fig.add_subplot(414)
            plt.show(block=False)


        else:
            self.ax1.plot(self.active_np_df[(self.n_max_indicators+self.frac_width):,1:5],  linewidth=1) # [:,1:4]
            #plt.title("OHCL_raw", fontsize=6)        

            self.ax2.plot(self.active_df.iloc[:,0:4],  linewidth=1) # [:,1:4]
            plt.title("OHCL scaled as inputs", fontsize=6)
            # add a subplot with no frame
            #plt.plot(self.active_df.iloc[:,37:39],  linewidth=1) # [:,1:4] # cusum positive, negative
            #plt.plot(self.active_df.iloc[:,39],  linewidth=1) # [:,1:4] # peaks and valleys
            #plt.plot(self.active_df.iloc[:,:-1:],  linewidth=1) # [:,1:4]
            self.ax3.plot(self.observation[:,:-1:],  linewidth=1) # [:,1:4]
            plt.title("All features", fontsize=6)

            self.ax4.plot(self.equity_curve[1:,2],  linewidth=1) # [:,1:4]
            plt.title("Exposure", fontsize=6)

            self.ax5.plot(self.equity_curve[1:,0:2],  linewidth=1) # [:,1:4]
            plt.title("Equity/balance", fontsize=6)
            #sns.lineplot(data=pd.DataFrame(self.equity_curve[1:,0:2]), x='0',y='1')


            self.ax6.plot(self.equity_curve[1:,3],  linewidth=1, color = 'grey') # [:,1:4]
            plt.title("Price Chart", fontsize=6)



            for i in range(1,x):
                exp = self.equity_curve[i,2]
                exp_prev = self.equity_curve[i-1,2]
                if exp != exp_prev:
                    if exp-exp_prev==1:                
                        self.ax7.scatter(x=i, y=self.equity_curve[i,3], color='green', marker ="^", s=4) 
                    elif exp-exp_prev==-1:                
                        self.ax7.scatter(x=i, y=self.equity_curve[i,3], color='red', marker ="v", s=4)
                    else: self.ax7.scatter(x=i, y=self.equity_curve[i,3], color='blue', marker ="s", s=5)
  
            plt.savefig("report"+".png")
            plt.draw() 
            plt.pause(0.0001)
            plt.cla()
            self.ax1.cla() 
            self.ax2.cla() 
            self.ax3.cla() 
            self.ax4.cla()
            self.ax5.cla()
            self.ax6.cla()
            #plt.close('all')

        
    def _plot_report(self):
    
        import matplotlib
        matplotlib.use('TkAgg') # <-- THIS MAKES IT FAST!
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')
        x= len(self.equity_curve) -1
        plt.ion()

        plt.figure(1) #fig_1 = plt.figure(1)
        ax1=plt.subplot(4,3,1)        
        plt.plot(self.active_np_df[(self.n_max_indicators+self.frac_width):,1:5],  linewidth=1) # [:,1:4]
        plt.title("OHCL_raw", fontsize=6)        

        ax2=plt.subplot(432)        
        plt.plot(self.active_df.iloc[:,0:4],  linewidth=1) # [:,1:4]
        plt.title("OHCL scaled as inputs", fontsize=6)
        # add a subplot with no frame
        ax3=plt.subplot(433)
        #plt.plot(self.active_df.iloc[:,37:39],  linewidth=1) # [:,1:4] # cusum positive, negative
        #plt.plot(self.active_df.iloc[:,39],  linewidth=1) # [:,1:4] # peaks and valleys
        #plt.plot(self.active_df.iloc[:,:-1:],  linewidth=1) # [:,1:4]
        plt.plot(self.observation[:,:-1:],  linewidth=1) # [:,1:4]
        plt.title("All features", fontsize=6)

        ax4=plt.subplot(412)        
        plt.plot(self.equity_curve[1:,2],  linewidth=1) # [:,1:4]
        plt.title("Exposure", fontsize=6)

        ax5=plt.subplot(413)
        plt.plot(self.equity_curve[1:,0:2],  linewidth=1) # [:,1:4]
        plt.title("Equity/balance", fontsize=6)
        #sns.lineplot(data=pd.DataFrame(self.equity_curve[1:,0:2]), x='0',y='1')

        ax6=plt.subplot(414) 
        plt.plot(self.equity_curve[1:,3],  linewidth=1, color = 'grey') # [:,1:4]
        plt.title("Price Chart", fontsize=6)

        ax6.grid()
        #ax7=plt.subplot(414)
        #for i in range(1,x):
        #    exp = self.equity_curve[i,2]
        #    exp_prev = self.equity_curve[i-1,2]
        #    if exp != exp_prev:
        #        if exp-exp_prev==1:                
        #            ax7.scatter(x=i, y=self.equity_curve[i,3], color='green', marker ="^", s=4) 
        #        elif exp-exp_prev==-1:                
        #            ax7.scatter(x=i, y=self.equity_curve[i,3], color='red', marker ="v", s=4)
        #        else: ax7.scatter(x=i, y=self.equity_curve[i,3], color='blue', marker ="s", s=5)
  
        plt.savefig("report"+".png")
        plt.draw() 
        plt.pause(0.0001)
        plt.cla()
        ax1.cla() 
        ax2.cla() 
        ax3.cla() 
        ax4.cla()
        ax5.cla()
        #ax6.cla()
        #plt.close('all')
        


    def _get_obs(self):

        #print("_get_obs")

        # desired and achivied goal for HER or DHER
        self.desired_goal =  self.desired_goal_calc(self.max_equity,self.max_balance) #self.reset_balance
        self.achieved_goal = self.achived_goal_calc(self.equity,self.balance)# 
 

        if self.ticks:
            if self.freqency_tickbar >= self._step:

                if self.tick_step<self.freqency_tickbar:

                    if self.tick_step==0:
                        self.last_active_df = self.df[self.frame_start +  self.bar_step + self.tick_step- self._step:
                                                self.frame_start + self.bar_step +self.tick_step ]#.astype(float)   #
                        self.last_active_df = get_tickbars_features(self.last_active_df, self.freqency_tickbar,diff_ticks=self.diff_ticks)
                        self.active_np_df = self.active_np_df[1:]
                        #print ('9221 active_df', self.active_np_df.shape)
                    else:
                        self.last_active_df = self.df[self.frame_start +  self.bar_step + self.tick_step- self._step:
                                        self.frame_start +  self.bar_step +self.tick_step ]#.astype(float)   # pd dataframe

                        self.last_active_df = get_tickbars_features(self.last_active_df, self.freqency_tickbar, diff_ticks=self.diff_ticks) # np array   
                        self.active_np_df = self.active_np_df[:-1]

                    self.active_np_df = np.append(self.active_np_df, self.last_active_df, axis=0)

    

            else: print('Error: step must be lower than freqency_tickbar - change settings')

        else : # OHCL data witout ticks and mt4
            #if self.mt4: self.last_active_df = self.df
            if self.mt4: 
                self.active_df = self.df.loc[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  ) +  self.ohlc_step :
                            self.frame_start +  self.ohlc_step ]                
                #self.active_df = self.df#[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  ) +  self.ohlc_step :  self.frame_start +  self.ohlc_step ]
                #print('67322 active_df.shape',self.active_df.shape)
            else:
                self.active_df = self.df[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  ) +  self.ohlc_step :
                            self.frame_start +  self.ohlc_step ]                

            self.active_np_df = get_ohlc_features(self.active_df) # np array

            #print('6358 self.active_np_df',self.active_np_df)

                #self.active_df = self.df[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  ) +  self.ohlc_step :
                #            self.frame_start +  self.ohlc_step ]
                #self.active_np_df = get_ohlc_features(self.active_df) # np array
        


        self.active_df = get_candles_features(self.active_np_df, frac_diff=self.frac_diff, fast_frac_diff = self.fast_frac_diff,d=self.d,thres =self.thres, log = self.log ,diff=self.diff) # pandas output
        self.active_df = self.active_df[-self.lookback:].reset_index(drop=True)
        #print ('2233',self.active_df.shape)
  
        obs = generate_obs(self.active_df)
        #print ('9854obs.shape',obs.shape)
        self._make_statement()

        profit_plus=0
        profit_minus=0
        if self.floating_profits>0:profit_plus=self.floating_profits/5000
        if self.floating_profits<0:profit_minus=-self.floating_profits/5000
        if profit_plus>1:profit_plus=1
        if profit_minus>1:profit_minus=1
        Profits=(self.balance-self.initial_balance_base)/self.balancex100
        self.account_history = np.append(self.account_history, [

            #[self.equity/self.balancex100],
            #[self.balance/self.balancex100], #[self.freemargin],
            #[self.max_equity/self.balancex100], #[self.floating_profits],
            #[self.min_equity/self.balancex100],
            #[(self.equity-self.prev_equity)/self.Nlot],
            #[(self.balance-self.prev_balance)/self.Nlot],
            #[profit_plus],#[self.floating_profits/1000], #self.achieved_goal self.desired_goal
            #[profit_minus],#[self.floating_profits/1000], #
            #[self.equity/self.balance-0.5],#[self.floating_profits/1000], #
            ##[self.freemargin/self.balancex100],
            [self.longs_floating/self.Max_allowed],          
            [self.shorts_floating/self.Max_allowed],          
            [self.floating_profits/5000],          
            [Profits]          #
            #[self.equity/self.balance-0.5],          
            #[self.equity/self.prev_equity-0.5]          
            #[(self.exposure+self.Max_allowed)/(2*self.Max_allowed)] 
            #[0],#self.floating_profits/10000]#self.equity/self.balance-1]#, [0]#, [0], [0], [0], [0]         
        ], axis=1)

        self.scaled_history =self.account_history  #scaler.fit_transform(self.account_history) #self.account_history#self.
        #print('424',self.scaled_history[ :,-1-self.lookback:-1: ])
        #print ('self.scaled_histor',self.scaled_history[ :,-self.lookback:].shape)
        #print ('obs',obs.shape)
        obs = np.append(obs,
            self.scaled_history[ :,-self.lookback:],
            axis=0).T
        self.observation = obs.copy()
        obs = np.resize(obs,obs.size) 
            #print ('3334444',obs.shape)
        
        
        if self.OBS_SPACE_GOALENV: 

            obs = OrderedDict([
                ('observation', obs),
                ('achieved_goal', self.achieved_goal.copy()), #np.array([self.balance])), #
                ('desired_goal', self.desired_goal.copy())
            ])       
        
        #print ('obs',obs)
                
        return obs

    

    def _reset_session(self):

        self.longs_floating = 0
        self.shorts_floating = 0
        self.exposure = 0
        
        if self.ticks:
            #if self.mt4:
            #    self.wait_for_sMt4()


            #self.active_np_df = self.df[self.frame_start - (self.lookback  + self.n_max_indicators +1) * self.freqency_tickbar-2:
            #                         self.frame_start  ].astype(float)  # + self._step_obs
            #self.active_np_df = get_tickbars_features(self.active_np_df, self.freqency_tickbar)

            #print ('11111', self.active_np_df.shape)


            self.active_np_df = self.df[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width ) * self.freqency_tickbar + self.bar_step +  self.tick_step :
                            self.frame_start + self.bar_step + self.tick_step]#.astype(float)  #
            self.active_np_df = get_tickbars_features(self.active_np_df, self.freqency_tickbar,diff_ticks=self.diff_ticks) # np array
            print ('7421 active_df', self.active_np_df.shape)
        
        else:
            if self.mt4:
                while len(self.df)<1 :
                    #time.sleep(2)
                    print(' reset No data')
                    #print('reset data',self.df)

    
                self.active_df = self.df.loc[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width + self.diffI  ) +self.ohlc_step :  self.frame_start +self.ohlc_step]#.astype(float)  #
                self.active_np_df = get_ohlc_features(self.active_df) # np array
            else:

                self.active_np_df = self.df[self.frame_start - (self.lookback  + self.n_max_indicators  + self.frac_width ) :
                                self.frame_start ]#.astype(float)  #
                self.active_np_df = get_ohlc_features(self.active_np_df) # np array


        #print('67859',self.active_np_df )


    def reset(self): 
             
        #print("RESET")
        
        if self.simple_report_in_learn: print('########RESET###########')


        if self.step_count ==0:


            self.balance = self.initial_balance_base
            self.equity = self.initial_balance_base
            self.prev_equity = self.initial_balance_base
            self.prev_balance = self.initial_balance_base
            self.freemargin = self.initial_balance_base
            self.used_margin = 0
            self.max_equity = self.initial_balance_base
            self.max_balance = self.initial_balance_base
            self.min_equity = self.initial_balance_base
            self.min_balance = self.initial_balance_base
            self.floating_profits = 0
            self.trades_counter = 0
            self.exposure = 0


            self.prev_equity = self.initial_balance_base

            self.desired_goal = np.array([(self.initial_balance_base)/self.balancex100])
            self.prev_achieved_goal = 0

            self._reset_session()

            profit_plus=0
            profit_minus=0
            if self.floating_profits>0:profit_plus=self.floating_profits/10000
            if self.floating_profits<0:profit_minus=-self.floating_profits/10000


            self.account_history = np.repeat([

            #[self.equity/self.balancex100],
            #[self.balance/self.balancex100], #[self.freemargin],
            #[self.max_equity/self.balancex100], #[self.floating_profits],
            #[self.min_equity/self.balancex100],
            #[(self.equity-self.prev_equity)/self.Nlot],
            #[(self.balance-self.prev_balance)/self.Nlot],
            #[profit_minus],#[self.floating_profits/1000], #
            [0 ],#[self.floating_profits/1000], #self.achieved_goal self.desired_goal
            [0],#[self.floating_profits/1000], #
            ##[self.freemargin/self.balancex100],
            [0],          
            [0]          
            #[(self.exposure+self.Max_allowed)/(2*self.Max_allowed)] 
            #[0],#self.floating_profits/10000]#self.equity/self.balance-1]#, [0]#, [0], [0], [0], [0]   

            ], self.lookback-1 , axis=1)

            account_longs = np.zeros((self.Max_allowed,1))
            self.account_longs = np.repeat(account_longs, self.lookback , axis=1)

            account_longs_lots = np.zeros((self.Max_allowed,1))
            self.account_longs_lots = np.repeat(account_longs_lots, self.lookback , axis=1)

            account_longs_ticket = np.zeros((self.Max_allowed,1))
            self.account_longs_ticket = np.repeat(account_longs_ticket, self.lookback , axis=1)


            account_shorts = np.zeros((self.Max_allowed, 1))
            self.account_shorts = np.repeat(account_shorts, self.lookback , axis=1)

            account_shorts_lots = np.zeros((self.Max_allowed, 1))
            self.account_shorts_lots = np.repeat(account_shorts_lots, self.lookback , axis=1)

            account_shorts_ticket = np.zeros((self.Max_allowed, 1))
            self.account_shorts_ticket = np.repeat(account_shorts_ticket, self.lookback , axis=1)
        
        self.dher_count = 1
        self.reset_balance=self.balance
        self.reset_equity=self.equity
        #self.prev_balance = self.balance
        #self.prev_equity = self.equity
        self.max_balance=self.balance
        self.max_equity=self.equity
        
        #print('self.reset_balance',self.reset_balance)
        obs =self._get_obs()
        return obs
        #return super(FxEnv, self).reset()

        
    def _get_current_time_tick(self):
        if self.mt4 and self.ticks: return self.df_raw.iloc[-1,0]#[self.frame_start + self.inter_step]
        elif self.ticks: return self.df['Timestamp'].values[self.frame_start + self.bar_step + self.tick_step]
        elif self.mt4 and not self.ticks:  return self.df['Time'].values[self.frame_start + self.ohlc_step]  # add spred




    def _get_current_Bid(self):
        if self.mt4 and self.ticks: return self.df_raw.iloc[-1,1]#[self.frame_start + self.inter_step]
        elif not self.mt4 and self.ticks: return self.df['Bid price'].values[self.frame_start + self.bar_step + self.tick_step].astype(float)
        elif self.mt4 and not self.ticks :  return self.df.loc[(self.frame_start + self.ohlc_step)]["Close"].astype(float)  # add spred
        elif not self.mt4 and not self.ticks :  return self.df['Close'].values[self.frame_start + self.ohlc_step].astype(float) 

    def _get_current_Ask(self):
        if self.mt4 and self.ticks : return self.df_raw.iloc[-1,2]#.values[self.frame_start + self.inter_step]
        elif not self.mt4 and self.ticks: return self.df['Ask price'].values[self.frame_start + self.bar_step + self.tick_step].astype(float)
        elif self.mt4 and not self.ticks:  return self.df.loc[(self.frame_start + self.ohlc_step)]["Close"].astype(float) + self.spread
        elif not self.mt4 and not self.ticks:  return self.df['Close'].values[self.frame_start + self.ohlc_step].astype(float) + self.spread
        
        ([0],  ['Country'])


    def close_one_short(self):
        #i = 0   #FIFO
        #LIFO
        if self.temp_account_shorts[self.Max_allowed-1]!=0: i=self.Max_allowed-1
        else:
            j = 0
            for j in range(self.Max_allowed):
                if self.temp_account_shorts[j]==0:break               
            i = j-1
        #for i in range(self.Max_allowed):
        if self.temp_account_shorts[i]!=0:
                    

            if self.mt4:              

                Ticket,ClosePrice,err = self._sMt4.OrderClose(ticket=self.temp_account_shorts_ticket[i], lots=0, price=0)

                
                if Ticket<0:
                    print('_sMt4 dont sent CLOSE SHORTS orders')  
                    print  ('_ret = ',_ret) 
                    profit = 0
                    lots1 = 0
                else:
                    price = ClosePrice
                    print ('slipage=',  self.current_Ask-price)
                    profit = self.temp_account_shorts[i] -price
                    lots1 = self.temp_account_shorts_lots[i]
                    self.temp_account_shorts[i] = 0  # removing  short
                    self.temp_account_shorts_lots[i] = 0  # removing  short
                    self.temp_account_shorts_ticket[i] = 0  # removing  short
                    self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                    self.balance = round(self.balance, 2)
                    self.exposure += lots1
                    self.shorts_floating -= 1
            else:
                price = self.current_Ask
                profit = self.temp_account_shorts[i] - price
                lots1 = self.temp_account_shorts_lots[i]
                self.temp_account_shorts[i] = 0  # removing  short
                self.temp_account_shorts_lots[i] = 0  # removing  short
                self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                self.balance = round(self.balance, 2)
                self.exposure += lots1
                self.shorts_floating -= 1


                    
            if self.simple_report_in_learn: print  ('.............Close Short...............', i, profit, lots1)




    def close_all_short(self):

  
        i = 0   

        for i in range(self.Max_allowed):
            if self.temp_account_shorts[i]!=0:
                    

                if self.mt4:              

                    Ticket,ClosePrice,err = self._sMt4.OrderClose(ticket=self.temp_account_shorts_ticket[i], lots=0, price=0)
                  
                
                    if Ticket <0:
                        print('_sMt4 dont sent CLOSE SHORTS orders')  
                        print  ('_ret = ',err) 
                        profit = 0
                        lots1 = 0
                    else:
                        price = ClosePrice
                        print ('slipage=',  self.current_Ask-price)
                        profit = self.temp_account_shorts[i] -price
                        lots1 = self.temp_account_shorts_lots[i]
                        self.temp_account_shorts[i] = 0  # removing  short
                        self.temp_account_shorts_lots[i] = 0  # removing  short
                        self.temp_account_shorts_ticket[i] = 0  # removing  short
                        self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                        self.balance = round(self.balance, 2)
                        self.exposure += lots1
                        self.shorts_floating -= 1

                else:
                    price = self.current_Ask
                    profit = self.temp_account_shorts[i] - price
                    lots1 = self.temp_account_shorts_lots[i]
                    self.temp_account_shorts[i] = 0  # removing  short
                    self.temp_account_shorts_lots[i] = 0  # removing  short
                    self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                    self.balance = round(self.balance, 2)
                    self.exposure += lots1
                    self.shorts_floating -= 1


                    
                if self.simple_report_in_learn: print  ('.............Close All Short...............', i, profit, lots1)



    def close_one_long(self):

        # last in first close
        #i = 0   
        if self.temp_account_longs[self.Max_allowed-1]!=0: i=self.Max_allowed-1
        else:
            j = 0
            for j in range(self.Max_allowed):
                if self.temp_account_longs[j]==0:break               
            i = j-1
        
        if self.temp_account_longs[i]!=0:
                    

            if self.mt4:              

                Ticket,ClosePrice,err = self._sMt4.OrderClose(ticket=self.temp_account_longs_ticket[i], lots=0, price=0)

                
                if Ticket <0:
                    print('_sMt4 dont sent CLOSE LONGS orders')  
                    print  ('_ret = ',err)
                    profit = 0
                    lots1 = 0
                else:
                    price = ClosePrice
                    print ('slipage=', price-self.current_Bid)
                    profit = price - self.temp_account_longs[i]
                    lots1 = self.temp_account_longs_lots[i]
                    self.temp_account_longs[i] = 0  # removing  long
                    self.temp_account_longs_lots[i] = 0  # removing  long
                    self.temp_account_longs_ticket[i] = 0  # removing  long
                    self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                    self.balance = round(self.balance, 2)
                    self.exposure -= lots1
                    self.longs_floating -= 1
            else:
                price = self.current_Bid
                profit = price - self.temp_account_longs[i]
                lots1 = self.temp_account_longs_lots[i]
                self.temp_account_longs[i] = 0  # removing  long
                self.temp_account_longs_lots[i] = 0  # removing  long
                self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                self.balance = round(self.balance, 2)
                self.exposure -= lots1
                self.longs_floating -= 1


                    
            if self.simple_report_in_learn: print  ('.............Close 1 Long...............', i, profit, lots1)



    def close_all_long(self):


        i = 0
        for i in range(self.Max_allowed):
            if self.temp_account_longs[i]!=0:
                    

                if self.mt4:  
                    
                    Ticket,ClosePrice,err = self._sMt4.OrderClose(ticket=self.temp_account_longs_ticket[i], lots=0, price=0)

                    
                
                    if Ticket < 0:
                        print('_sMt4 dont sent CLOSE LONGS orders')  
                        print  ('_ret = ',err)
                        profit = 0
                        lots1 = 0
                    else:
                        price = ClosePrice
                        print ('slipage=', price-self.current_Bid)
                        profit = price - self.temp_account_longs[i]
                        lots1 = self.temp_account_longs_lots[i]
                        self.temp_account_longs[i] = 0  # removing  long
                        self.temp_account_longs_lots[i] = 0  # removing  long
                        self.temp_account_longs_ticket[i] = 0  # removing  long
                        self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                        self.balance = round(self.balance, 2)
                        self.exposure -= lots1
                        self.longs_floating -= 1
                else:
                    price = self.current_Bid
                    profit = price - self.temp_account_longs[i]
                    lots1 = self.temp_account_longs_lots[i]
                    self.temp_account_longs[i] = 0  # removing  long
                    self.temp_account_longs_lots[i] = 0  # removing  long
                    self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
                    self.balance = round(self.balance, 2)
                    self.exposure -= lots1
                    self.longs_floating -= 1


                    
                if self.simple_report_in_learn: print  ('.............Close All Long...............', i, profit, lots1)
        







    def _take_action(self, action_type, lots, long_to_close, short_to_close):
        #print('_take_action')

        self.prev_balance1 = self.prev_balance
        self.prev_balance=self.balance
        TrueLeverage = (( self.exposure ) *self.Nlot)/self.equity
        if self.simple_report_in_learn: print ('TrueLeverage = ',TrueLeverage )        
        #Margin Requirement = Current Price × Units Traded × Margin
        self.req_margin = self.Nlot / self.leverage * self.current_Bid * self.Quote_Home_Currency # 


        if self.equity<=self.used_margin:
            print('MARGIN CALL')
            print ('CATS: You are on the way to destruction.')

        while self.margin_level < self.stop_out_level:  # self.margin_level = self.equity/self.used_margin *100 
            print('STOP OUT - closing trade')
            print('Cats: You have no chance to survive make your time.') 
            print ('Cats: HA HA HA HA .... ')            
            self.close_one_long()                
            self.close_one_short()                
            self.cleanup_stack()
            self._get_obs()



            ########################
        if action_type == 0  : # Long
             
            
            #if self.temp_account_shorts[short_to_close] != 0: self.close_all_short()

            if self.temp_account_shorts[0] != 0 and self.hedge==False: self.close_one_short() # No hedge
            else:
                if self.longs_floating  < self.Max_allowed:
           
                    if self.used_margin +  lots * self.req_margin < self.equity:

                        if self.temp_account_longs[[self.longs_floating]]==0:

                            #self.close_all_short()
                            if self.mt4:
                            
                                # Send instruction to MetaTrader
                                #_ret = self._execution._execute_(self._default_order,
                                #                            self._verbose,
                                #                            self._delay,
                                #                            self._wbreak)
                                Ticket,OpenPrice,err = self._sMt4.OrderSend(symbol=self.mt4_symbol, type= 'OP_BUY', lots=0.01, price= 0, SL=0, TP=0, magic=self.magic,comment ='Py_to_MT')
                                print (Ticket) #-1,-1,'ok'# 
                                if Ticket < 0:
                                    print('_sMt4 dont sent LONG order')
                                    lots=0
                                else:
                                    print  ('_ret = ',Ticket)
                                    self.temp_account_longs[[self.longs_floating]] = OpenPrice
                                    print ('slipage=', self.current_Ask - OpenPrice)
                                    self.temp_account_longs_lots[self.longs_floating] = lots # adding new  long lots
                                    self.temp_account_longs_ticket[self.longs_floating] = Ticket # adding new ticket
                                    self.trades_counter += 1
                                    self.exposure += lots
                                    self.longs_floating += 1                        
                        
                            else:
                                self.temp_account_longs[[self.longs_floating]] = self.current_Ask # adding new long price
                                self.temp_account_longs_lots[self.longs_floating] = lots # adding new 
                                self.trades_counter += 1
                                self.exposure += lots
                                self.longs_floating += 1
                                # self.action = 1
                        if self.simple_report_in_learn: print  ('.............LONG...............', lots, '............', self.current_Ask)
                        
                    else:
                        if self.simple_report_in_learn: print('................................not enough funds to open next trade....................................')
                else:
                    if self.simple_report_in_learn: print('................................Sent LONG but Max trades exceeded....................................')


        elif action_type == 1 : ##  Short

            #if self.temp_account_longs[long_to_close] != 0: self.close_all_long()
  
            
            if self.temp_account_longs[0] != 0  and self.hedge==False:self.close_one_long() # No hedge
            else:
                if self.shorts_floating  < self.Max_allowed:
                    if self.used_margin + lots * self.req_margin < self.equity:

                        if self.temp_account_shorts[[self.shorts_floating]] == 0:
                            #self.close_all_long()
                            if self.mt4:
                                Ticket,OpenPrice,err = self._sMt4.OrderSend(symbol=self.mt4_symbol, type= 'OP_SELL', lots=0.01, price= 0, SL=0, TP=0, magic=self.magic,comment ='Py_to_MT')
                                # -1,-1,'ok'# 
                                print (Ticket)
                                if Ticket < 0:
                                    print('_sMt4 dont sent SHORT order')
                                    lots=0
                                else:
                                    print  ('_ret = ',Ticket)
                                    self.temp_account_shorts[[self.shorts_floating]] = OpenPrice
                                    print ('slipage=',  OpenPrice-self.current_Bid)
                                    self.temp_account_shorts_lots[self.shorts_floating] = lots  # adding new short 
                                    self.temp_account_shorts_ticket[self.shorts_floating] = Ticket # adding new ticket
                                    self.trades_counter += 1
                                    self.exposure -= lots
                                    self.shorts_floating += 1
                                    # self.action = -1

                            else:
                                self.temp_account_shorts[[self.shorts_floating]] = self.current_Bid  # adding new long price
                                self.temp_account_shorts_lots[self.shorts_floating] = lots  # adding new long 
                                self.trades_counter += 1
                                self.exposure -= lots
                                self.shorts_floating += 1
                            if self.simple_report_in_learn: print  ('.............SHORT...............', lots, '............', self.current_Bid)

                    else:
                        if self.simple_report_in_learn: print('................................not enough funds to open next trade....................................')
                else:
                    if self.simple_report_in_learn: print('................................Sent SHORT but Max trades exceeded....................................')




        elif action_type == 2: #  Hold


            if self.simple_report_in_learn: print  ('.............HOLD...............', self.balance, '............',self.equity)



        elif action_type == 3 : #  Close All
        
            if self.temp_account_longs[long_to_close] != 0: self.close_all_long()
            if self.temp_account_shorts[short_to_close] != 0: self.close_all_short()
            if self.simple_report_in_learn: print  ('.............CLOSE ALL...............', self.balance, '............',self.equity)


        elif action_type == 4 and self.temp_account_longs[long_to_close] !=0: #  Close Longs
            #if self.exposure > 0 and self.req_margins <= self.exposure:

            self.close_all_long()

        elif action_type == 5 and self.temp_account_shorts[short_to_close] != 0: #  Close Short

            self.close_all_short() 
            
        self.cleanup_stack()
           
        

    def cleanup_stack(self):

        for x in range(0, self.Max_allowed):
            if self.temp_account_shorts[x] == 0:
                if x != self.Max_allowed - 1:
                    self.temp_account_shorts[x] = self.temp_account_shorts[x + 1]  # cleaning stack
                    self.temp_account_shorts[x + 1] = 0
                    self.temp_account_shorts_lots[x] = self.temp_account_shorts_lots[x + 1]  # cleaning stack
                    self.temp_account_shorts_lots[x + 1] =0

                    self.temp_account_shorts_ticket[x] = self.temp_account_shorts_ticket[x + 1]  # cleaning stack
                    self.temp_account_shorts_ticket[x + 1] =0
                    #print('self.temp_account_shorts',self.temp_account_shorts)
        #print('3self.temp_account_shorts', self.temp_account_shorts)

        for x in range(0, self.Max_allowed):
            if self.temp_account_longs[x] == 0:
                if x != self.Max_allowed - 1:
                    self.temp_account_longs[x] = self.temp_account_longs[x + 1]  # cleaning stack
                    self.temp_account_longs[x + 1] = 0
                    self.temp_account_longs_lots[x] = self.temp_account_longs_lots[x + 1]  # cleaning stack
                    self.temp_account_longs_lots[x + 1]=0
                    self.temp_account_longs_ticket[x] = self.temp_account_longs_ticket[x + 1]  # cleaning stack
                    self.temp_account_longs_ticket[x + 1]=0
                    #print('self.temp_account_longs',self.temp_account_longs)


        self.account_shorts = np.append(self.account_shorts, self.temp_account_shorts.reshape((-1, 1)), axis=1)
        self.account_shorts_lots = np.append(self.account_shorts_lots, self.temp_account_shorts_lots.reshape((-1, 1)), axis=1)
        self.account_shorts_ticket = np.append(self.account_shorts_ticket, self.temp_account_shorts_ticket.reshape((-1, 1)), axis=1)


        self.account_longs = np.append(self.account_longs, self.temp_account_longs.reshape((-1, 1)), axis=1)
        self.account_longs_lots = np.append(self.account_longs_lots, self.temp_account_longs_lots.reshape((-1, 1)), axis=1)
        self.account_longs_ticket = np.append(self.account_longs_ticket, self.temp_account_longs_ticket.reshape((-1, 1)), axis=1)

    def _make_step(self) :
        #print("make_Step")
        if self.ticks:
            if  self.tick_step + self._step >= self.freqency_tickbar: 
                self.tick_step =0
                self.bar_step +=self.freqency_tickbar
            else:
                self.tick_step += self._step
                self.bar_step=self.bar_step 

            if self.mt4: self.wait_for_sMt4()


        else:
            #self.ohlc_step+=1
            if self.mt4: self.wait_for_sMt4()
            

            


    def check_SLTP(self) :
         
        if self.ticks:
                if self.tick_step==0:
                    ticks = self.df[self.frame_start +  self.bar_step + self.tick_step - self._step:
                                            self.frame_start +  self.bar_step + self.tick_step]#.astype(float)   #
                else:
                    ticks = self.df[self.frame_start +  self.bar_step + self.tick_step - self._step:
                                    self.frame_start +  self.bar_step + self.tick_step ]#.astype(float)   # pd dataframe


                ticks=ticks.dropna().reset_index(drop=True).values

                for i in range(0,len(ticks)):
                    Price = ticks[i, 1]#.astype(float)
                    for x in range(0, self.Max_allowed):
                                Orderprice1 = self.account_longs[x, -1]                            
                                if Orderprice1!=0:
                                        if Price<=Orderprice1 - self.SL  or   Price>= Orderprice1 + self.TP  :
                                            self.CloseLongAtPrice(x,Price)
                                        

                                Orderprice2 = self.account_shorts[x, -1]
                                if Orderprice2!=0:
                                    if Price>=Orderprice2 + self.SL or Price<=Orderprice2 - self.TP   :
                                        self.CloseShortAtPrice(x,Price) 
                self.cleanup_stack() 
       
    def CloseLongAtPrice(self,i,Price):
        print('CloseLongAtPrice',Price)


               
        if self.temp_account_longs[i]!=0:                  

            
            
            price = Price
            profit = price - self.temp_account_longs[i]
            lots1 = self.temp_account_longs_lots[i]
            self.temp_account_longs[i] = 0  # removing  long
            self.temp_account_longs_lots[i] = 0  # removing  long
            self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
            self.balance = round(self.balance, 2)
            self.exposure -= lots1
            self.longs_floating -= 1


                    
            if self.simple_report_in_learn: print  ('.............Close 1 Long...............', i, profit, lots1)


    def CloseShortAtPrice(self,i,Price):
        print('CloseShortAtPrice',Price)

        
        if self.temp_account_shorts[i]!=0:                 
            

            
            price = Price
            profit = self.temp_account_shorts[i] - price
            lots1 = self.temp_account_shorts_lots[i]
            self.temp_account_shorts[i] = 0  # removing  short
            self.temp_account_shorts_lots[i] = 0  # removing  short
            self.balance = self.balance + (profit  * lots1 * self.Nlot) - self.commission * lots1
            self.balance = round(self.balance, 2)
            self.exposure += lots1
            self.shorts_floating -= 1


                    
            if self.simple_report_in_learn: print  ('.............Close Short...............', i, profit, lots1)


    def step(self, action):
        #print("step")
        self.step_count+=1
        self.dher_count += 1
        #print ('step')
        #assert self.action_space.contains(action)
        if self.simple_report_in_learn: print ('action',action)

        ######################

        
        self.temp_account_shorts = (self.account_shorts[:, -1])  # copy last row !!!!!!!!!!!!!!
        self.temp_account_shorts_lots = (self.account_shorts_lots[:, -1])  # copy last row
        self.temp_account_shorts_ticket = (self.account_shorts_ticket[:, -1])  # copy last row

        self.temp_account_longs = (self.account_longs[:, -1])  # copy last row
        self.temp_account_longs_lots = (self.account_longs_lots[:, -1])  # copy last row
        self.temp_account_longs_ticket = (self.account_longs_ticket[:, -1])  # copy last row

        if self.ACTION_SPACE_MULTI:
            action_type = action[0]
            lots = action[1]
            long_to_close = action[2]
            short_to_close = action[3]

        elif self.ACTION_SPACE_BOX:
            #action_type = int(abs(np.asscalar((action+self.nA)/2)))
            #action_type = int(abs(np.asscalar(action[1,0])))
            a = np.asscalar(action[0])
            #print ('a',a)
            b = np.asscalar(action[1])
            #print ('b',b)
            if a >0 and b>0: action_type=0
            if a <0 and b>0: action_type=1
            if a >0 and b<0: action_type=2
            if a <=0 and b<=0: action_type=3
            lots = 1
            #long_to_close = 0
            #short_to_close = 0

        else:
                nL=1
                if self.lot_sizing: nL=10
                actions = (self.nA * nL * 1 * 1)
            #if action <= actions/self.nA:
             #   self.action_type = 0

                #print(self.action_type)
                a=0
                while a<=self.nA:
                    action_type = a
                    a += 1
                    if action  < actions/self.nA*a:
                        #print(action_type)
                        
                        break
                lots = 1

                if self.lot_sizing:
                    b=0
                    while b<=10:
                        lots = b
                        b += 1
                        if action -actions/self.nA*action_type < actions/self.nA/10*b:
                            #print(lots)
                            break
                #c=0
                #while c<=1:
                #    long_to_close = c
                #    c += 1
                #    if action  -actions/self.nA*action_type -actions/self.nA/10*lots < actions/self.nA/10/self.Max_allowed*c:
                #        print(long_to_close)
                #        break
                #d=0
                #while d<=1:
                #    short_to_close = d
                #    d += 1
                #    if action -actions/self.nA*action_type  - actions/self.nA/10*lots - actions/self.nA/10/self.Max_allowed*long_to_close < actions/self.nA/10/self.Max_allowed/self.Max_allowed*d :
                #        print(short_to_close)
                #        break

    
        if action_type == 0: self.longs=self.longs+1
        if action_type == 1: self.shorts=self.shorts+1

        self.action_typ=action_type 
        if self.simple_report_in_learn: 
            print('action_type=',action_type ) 
            print('lots =',lots ) 
        


        
        #print('action_type, lots',action_type, lots)
        
        self._take_action(action_type, lots,0,0) 
        self._make_step()
        self.ohlc_step+=1 # ZA MAKE_STEP
        self.check_SLTP() # make step itd
        obs = self._get_obs()

        
        
        #print (obs)
        done = False
        if self.balance < self.prev_balance :
        #if self.equity > self.prev_equity :
            self.done1=self.done1+1
            

        if self.test:
            if self.OBS_SPACE_GOALENV:
                done = self._is_success(self.achieved_goal, self.desired_goal)
                if self.dher_count % self.lookback==0: done=True 

    
            else:
                if self.done1>=1 :                 
                    done=True
                    self.done1=0
                 
                #if self.step_count % self.episod_steps==0: done=True 
                #else: done=False
                #if self.exposure==0: done=True
        else: 
            if self.OBS_SPACE_GOALENV:
                done = self._is_success(self.achieved_goal, self.desired_goal)
                if self.dher_count % self.lookback==0: done=True 

                    
                    
            else:
                if self.done1>=1 :                 
                    done=True
                    self.done1=0
                  
                #if self.step_count % self.episod_steps==0: done=True 
                #else: done=False
                #if self.exposure==0: done=True


        
        if self.OBS_SPACE_GOALENV:            
            info = {
                'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
            }

            if self.simple_report_in_learn: print('info',info['is_success'])
            reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info['is_success'])          
            

        else: 
            reward = self.reward() # Clasic reward
            #reward = self.compute_reward(self.achieved_goal, self.desired_goal)   # reward for DHER       
            #done = self._is_success(self.achieved_goal, self.desired_goal) # done for DHER 
            #if self.dher_count % self.lookback==0: done=True 
            info = {'is_bancrucy': done}

        
        self.prev_reward_sum = self.reward_sum
        self.reward_sum = self.reward_sum + reward
        #print('reward sum=',self.reward_sum)
        if done and self.simple_report_in_learn: print ('...........DONE ............... ',done)

        if done :

            self.reward_sum=0
            self.done1=0
            self.reset_balance=self.balance
            self.reset_equity=self.equity
            self.prev_balance = self.balance
            self.prev_equity = self.equity
            self.max_balance=self.balance
            self.max_equity=self.equity
            self.dher_count =0


            print('EPISODE Done')
           #self.dg =  self.desired_goal_calc(self.equity,self.balance) #self.desired_goal_calc(self.reward_sum,self.reward_sum)
        if self.simple_report_in_learn: print ('REWARD = ',reward) 


        ######################################self._make_step()  ###!!!


        if self.mt4==False:
            if (self.frame_start + self.bar_step + self.tick_step>= len(self.df) -  2*self.freqency_tickbar and self.mt4 == False and self.ticks) or (self.frame_start + self.ohlc_step >= len(self.df)-10  and  self.ticks==False) : # or (self.test==False and self.bar_step/self._step_obs >=100 and self.mt4 == False):

                self.balance =  self.balance + self.floating_profits # Closing all positions
                self.equity = self.balance

                account_longs = np.zeros((self.Max_allowed, 1))
                self.account_longs = np.repeat(account_longs, self.lookback , axis=1)

                account_longs_lots = np.zeros((self.Max_allowed, 1))
                self.account_longs_lots = np.repeat(account_longs_lots, self.lookback , axis=1)

                account_longs_ticket = np.zeros((self.Max_allowed, 1))
                self.account_longs_ticket = np.repeat(account_longs_ticket, self.lookback , axis=1)


                account_shorts = np.zeros((self.Max_allowed, 1))
                self.account_shorts = np.repeat(account_shorts, self.lookback , axis=1)

                account_shorts_lots = np.zeros((self.Max_allowed, 1))
                self.account_shorts_lots = np.repeat(account_shorts_lots, self.lookback , axis=1)

                account_shorts_ticket = np.zeros((self.Max_allowed, 1))
                self.account_shorts_ticket = np.repeat(account_shorts_ticket, self.lookback , axis=1)
            
                self.bar_step = 0
                self.tick_step = 0
                self.ohlc_step = 0
            

                print ('################STOP###########')
                #time.sleep(0)
                self._reset_session()  
            
        self.equity_curve_temp =  np.array([[self.equity,self.balance,self.exposure,self.current_Bid]], dtype=np.float)        
        self.equity_curve = np.append(self.equity_curve, self.equity_curve_temp, axis=0)
        #self._plot_report()

        return obs, reward, done, info
        #return obs, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

 

    def render(self, mode='human', **kwargs):
        if mode == 'system':

            print('Equity: ' + str(self.equity))
            print('Balance: ' + str(self.balance))
            #print('Free margin: ' + str(self.freemargin))
            #print('Used margin: ' + str(self.used_margin))
            print('Folating profits: ' + str(self.floating_profits))

            #print('Floating Longs: ' + str(self.longs_floating))
            #print('Floating Shorts: ' + str(self.shorts_floating))
            print('Exposure: ' + str(self.exposure))
            print('Action: ' + str(self.action_typ))
            print('Stats Longs/Shorts: ' + str(self.longs) +' vs '+ str(self.shorts)  )
            if self.ticks: print('-----------------------------'+str(self.df['Timestamp'].values[self.frame_start + self.bar_step + self.tick_step]) )
            if not self.ticks: print('-----------------------------'+str(self.df['Date'].values[self.frame_start + self.ohlc_step])+' | '
                                     +str(self.df['Timestamp'].values[self.frame_start + self.ohlc_step]))
             
            
        elif mode == 'simple':
            print('Equity: ' + str(self.equity))
            print('Balance: ' + str(self.balance))
            


        elif mode == 'human':
            
            if self.step_count % self.report_interval==0:
                self._plot_report()
                
            else:
                pass

    def close(self):                 
                
            self._get_obs()
            self.step_count=0
            self.bar_step = 0
            self.tick_step = 0
            self.ohlc_step=0
            np.savetxt("equity_curve.csv", self.equity_curve, delimiter=",")
            
            pass


    def ThreadingMT4(self, interval=0.03):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.int = interval
        
        self.que =queue.Queue()
        self.thread = threading.Thread(target=self.runMT4, kwargs={'frame_start': self.frame_start},args=() )

        self.thread.daemon = True                            # Daemonize thread
        self.thread.start()                                  # Start the execution
      

    def runMT4(self,frame_start=100):
        """ Method that runs forever """
        
        if self.ticks:
            print('runmt4')
            self._sMt4.Subscribe_BidAsk(symbol=self.mt4_symbol )
            print('Waiting for mt4 data in the background')
            self.dft = pd.DataFrame()
            # self._DB_sMt4[_symbol] = (_timestamp, float(_bid), float(_ask))

            while self._sMt4._DB_sMt4.empty :
                time.sleep(1)
                print('No data')
            self.dft = self._sMt4._DB_sMt4
            print('dft',self.dft)
        
            df_actual = self.dft
            dfprev=df_actual
            dfall = df_actual       
        
            while True :
            
                #df_actual = pd.DataFrame.from_dict(self._sMt4._DB_sMt4.values())
                df_actual = self._sMt4._DB_sMt4

                if df_actual.iloc[0,0]==dfprev.iloc[0,0]: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    dfall=dfall
                    dfprev=df_actual
                    #print('abcd',dfprev.iloc[0,0])
                else:
                        self.que.put(df_actual)
                        dfall = pd.concat([dfall, df_actual], ignore_index=True, axis=0)
                        dfprev=df_actual
                        self.df_raw = dfall.copy()
                        #print ('dfall',dfall.shape) 
                        #dfall.to_csv('E:/GBPUSD_thread.csv', sep = ";", index=False)        
                        #print (self.df_actual[-1]) 
                time.sleep(self.int)
        else:
            print('runMT4 Waiting for mt4 data in the background')
            self._sMt4.Subscribe_OHLC(symbol=self.mt4_symbol,timeframe=self.mt4_timeframe,bars=frame_start)
            while self._sMt4._DB_sMt4.empty :
                time.sleep(0.55)
                print('runMT4 No data')
            #self.df = self._sMt4._DB_sMt4
            #while True:  
            #    time.sleep(1)
            #    #print('runMT4 No data')
            #    if  self._sMt4._DB_sMt4['Timestamp'].values[-1]!=self.df['Timestamp'].values[-1] :

            #        self.df = self._sMt4._DB_sMt4.copy()
            #        self.df.index = self.df.index + 1
            #        #print('data',self.df)


            #self.df = self._sMt4._DB_sMt4.copy()

            #from sys import stdout
            #while self._sMt4._DB_sMt4.iloc[1,1]==self.df.iloc[1,1] or self._sMt4._DB_sMt4.empty:
            #    i=("Waiting for next bar")
            #    print(i)
            #    #stdout.write("\r%" % i)
            #    #stdout.flush()
            #    time.sleep(0.22)
            ##stdout.write("\n")
            #self.df=self._sMt4._DB_sMt4.copy()
            #self.df.index + self.ohlc_step
            #print(self.df)         



    def reward(self): 
        
        rew=0
        rew1=0
        rew2=0
        rew3=0

         
        #rew2=-100
        #rew = np.log(self.equity/self.balance)*100
        rew1 = np.log(self.equity/self.prev_equity)*100
        rew2 =  np.log(self.balance/self.prev_balance)*1000
        #self.prev_exposure=self.exposure
        #if rew!=0: rew2=0


        #if self.exposure==0: rew=-1
        #if self.balance>self.prev_balance: rew3=10
        #if self.balance>self.reset_balance: rew=100
        #if self.equity==self.prev_equity: rew3=-1
        #if self.equity<=self.balance: rew1=-2
        #rew3 =  np.log(self.balance/self.prev_balance)*10000

        r = rew +rew1 +rew2+rew3
       
        #print(self.step_count,'  Eq:',self.equity,'/Bal:',self.balance,'/Exposure:',self.exposure,'Rew:',r)
        return r*1
        


    def compute_reward(self, achieved_goal, desired_goal, info=False):
        
        if self.simple_report_in_learn: 
            print('achieved_goal_cr',achieved_goal)
            print('desired_goal_cr',desired_goal)
            print('_info_cr',info)       

        self.prev_exposure=self.exposure

        rew=-1
        rew1=0 #(self.balance-20000)/10000
        #ag = np.array([((self.equity)/1)/self.balancex100])              
        #rew1=goal_distance(ag, desired_goal) *1e4 # np.log(self.equity/self.prev_balance)*1000
        rew2= np.asscalar( self.goal_distance(achieved_goal, desired_goal))  # np.log(self.equity/self.prev_balance)*1000
        #if self.equity>self.reset_equity: rew2=-rew2
        #if self.balance>self.prev_balance: rew=10
        #if self.balance<self.prev_balance: rew=-2
        #if self.balance>self.reset_balance: rew=100
        #rew1 = np.log(self.balance/self.prev_balance)*10000
        #rew = (np.log(self.equity/self.prev_equity)*100)     
        r = rew + rew1 +rew2
        #print(self.step_count,'  Eq:',self.equity,'/Bal:',self.balance,'/Exposure:',self.exposure,'Rew:',r) #,achieved_goal,desired_goal

        return r




    def _is_success(self, achieved_goal, desired_goal):
        #if  achieved_goal[0] > desired_goal[0] : #
        if  achieved_goal[0] > desired_goal[0] : #
            
            self.success=self.success+1
            return True        
        else:             
            return False

        


    def desired_goal_calc(self, equity, balance):
        
        bal=(balance)/self.balancex100
        eq =(equity)/self.balancex100
        goal = np.array([bal])        
        goal1 = np.array([eq])        
        goal2 = np.array([eq,bal])        
        
        return goal

    def achived_goal_calc(self, equity, balance):
        
        bal=(balance)/self.balancex100
        eq =(equity)/self.balancex100
        goal = np.array([bal])        
        goal1 = np.array([eq])        
        goal2 = np.array([eq,bal])         
        
        return goal

    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        
        d = np.linalg.norm(goal_a - goal_b, axis=-1, ord=1) #, ord=1
        #if goal_a<goal_b: d=-d
        #return -( d== 0.0).astype(np.float32)/1 + d  *1e5 # 
        #return -( d <= 0.0).astype(np.float32)/1 # sparse reward
        #return -( d <= 0.0).astype(np.float32)/1 -d *1e4 # sparse reward
        #return -( d <= 0.0).astype(np.float32)/1 #+d *1e5 # sparse reward
        return -d *1e2
    def goal_distance1(self,goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        
        d = (goal_a - goal_b) #
        #return -( d== 0.0).astype(np.float32)/1 + d  *1e5 # 
        #return -( d <= 0.0).astype(np.float32)/1 # sparse reward
        return -d *1e4



