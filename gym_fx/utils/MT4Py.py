#https://realpython.com/python-sockets/
## https://www.mql5.com/en/blogs/post/706665
#!/usr/bin/env python3

import socket
import pandas as pd
import numpy as np
import io
import time
import threading
import threading

class ServerMT4:   
       
    def __init__(self,
                 ServerPort = 2540,
                 episod_steps = 50):
        self.PORT=ServerPort    # The port used by the server
        self.HOST = '127.0.0.1'  # The server's hostname or IP address
        self.reqtp = 1

        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.req:
        self.req=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.req.bind((self.HOST, self.PORT))
        self.req.listen()
        self.cmndonn, self.addr = self.req.accept()
        self._MarketData_Thread = Thread(target=self.ticks, args=())
        self._MarketData_Thread.start()           


    def ticks(self):

        with self.cmndonn:
            print('Connected by', self.addr)

            while self.reqtp:
                data = self.cmndonn.recv(16384)
                datastr = data.decode("ASCII")
                #datastr = io.BytesIO(data);
                if not data:
                    break
                print( datastr)
                self.cmndonn.send(data)

    def stop(self,stp=0):            
            if stp==0:self.reqtp=0
            else: self.reqtp=1

    def close(self):            
            self.req.close()

###Echo Server
#HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
#PORT = 23456        # Port to listen on (non-privileged ports are > 1023)

#with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#    s.bind((HOST, PORT))
#    s.listen()
#    conn, addr = s.accept()
#    with conn:
#        print('Connected by', addr)
#        dfa= pd.DataFrame()
#        while True:
#            import re 
#            data = conn.recv(1024)
#            #res = re.findall(r'\w+', data.decode("ASCII")) 
#            datastr = data.decode("ASCII")
#            #datastr = io.BytesIO(data);
#            #df = pd.read_csv(datastr)
            
#            if not data:
#                break
#            print( datastr)
#            conn.sendall(data)

###Echo Client
#!/usr/bin/env python3



#import socket
import asyncore
class ClientMT4(asyncore.dispatcher_with_send):   
       
    def __init__(self,
                 ServerPort = 23457,
                 #_PUSH_PORT = 23457,
                 verbose=False):
        self.serverPort=ServerPort    
        self._PUSH_PORT=ServerPort    
        self.HOST = '127.0.0.1'  # The server's hostname or IP address
        #PORT = self.serverPort        # The port used by the server
        self._DB_sMt4 = pd.DataFrame() #Database market data
        self.connect()
        self.rx="##"

    def __del__(self): 
        print('Destructor called, Employee deleted.')        
        MSG = bytes("close"+'\r\n', "ASCII")
        self.req.sendall(MSG)
        self.cmnd.sendall(MSG)
        self.req.close()
        self.cmnd.close()

    def connect(self):
            data=data1=""
            self.req=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #self.req.settimeout(30)
            try:
                self.req.connect((self.HOST, self.serverPort))
                data = self.req.recv(1024).decode("ASCII")
                #time.sleep(0.5)
            except:print("Can't connect to mt4 server")

            self.cmnd=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #self.cmnd.settimeout(30)
            try:
                self.cmnd.connect((self.HOST, self.serverPort))
                data1 = self.cmnd.recv(1024).decode("ASCII")
                #time.sleep(0.5)
            except:print("Can't connect to mt4 server")
                 
                 
            print("Connected", data,data1)
  

    def _rcmd(self):            
            #data="~~"
            #while data=="~~":                
            #    data = self.cmnd.recv(1024).decode("ASCII")   
            #    self.cmnd.close()
            #return data
            #    #print('comd',data)
            #    #time.sleep(0.5)
            while True:
                data = self.cmnd.recv(4096).decode("ASCII")
                data=data.split()
                if  data: break            
            return data[0]

    def _recive(self):
                data = self.req.recv(16384)#.decode("ASCII")
                #print('data',data)
                if data == b'NO DATA' :  pass 
                else:
                    data1 = io.BytesIO(data)                
                    #data1 = io.StringIO(data)               
                    #df = pd.read_csv(data1,delimiter ="|")#.stack(level=-1, dropna=True)
                    df= pd.read_table(data1, delimiter=self.rx, header=0, engine="python") #encoding="ASCII",
               
                self._DB_sMt4 = df  
                #print(df)
                #print(self._DB_sMt4)

    def Subscribe_BidAsk(self,symbol='EURUSD',last_ticks=10000):
        #self.cmndonnect()
        print('bidask')
        self._MarketData_Thread = threading.Thread(target=self._bidask, kwargs={'symbol': symbol,'last_ticks': last_ticks}, args=())
        #self._MarketData_Thread.daemon = True 
        self._MarketData_Thread.start()
        
    
    def _bidask(self,symbol='EURUSD',last_ticks=10000):
            print('bidask1')
            MSG='Subscribe_BidAsk/'+str(symbol) + "/last_ticks:" + str(last_ticks)
            MSG = bytes(MSG+'\r\n', "ASCII")
            
            while True:
                #print('as')
                self.req.sendall(MSG)
                print('send:',MSG)
                self._recive()
                
    def Subscribe_OHLC_thread(self,symbol='EURUSD',timeframe=5,bars=50):
        #self.cmndonnect()
        #print('OHLC')
        self._MarketData_Thread = threading.Thread(target=self._OHLC, kwargs={'symbol': symbol,'timeframe': timeframe,'bars': bars}, args=())
        #self._MarketData_Thread.daemon = True 
        self._MarketData_Thread.start()

    def Subscribe_OHLC(self,symbol='EURUSD',timeframe=5,bars=50):            
            #ret = "/"
            MSG = "Subscribe_OHLC:" + str(symbol) +   "/timeframe:" + str(timeframe) +  "/bars:" + str(bars)
            MSG = bytes(MSG+'\r\n', "ASCII")
            while True:
                self.req.sendall(MSG)
                self._recive()
                time.sleep(3)

                
            

    def OrderSend(self,symbol='EURUSD', type= 'OP_BUY', lots=0.01, price= 0, SL=0, TP=0, magic=12340,comment ='Python_to_MT'):
            
            ORD_TYPE=type
            MSG = "OrderSend:" + str(symbol) +  '/type:' + type + '/lots:' + str(lots) + '/price:' + str(price) + '/SL:' + str(SL) + '/TP:' + str(TP)+ '/magic:' + str(magic) + '/comm:'  + comment
            MSG = bytes(MSG+'\r\n', "ASCII")
            self.cmnd.sendall(MSG)
            while True:
                data = self.cmnd.recv(4096).decode("ASCII")
                if  data:
                    break            
            print(data)
            Ticket,OpenPrice,err = data.split(self.rx)
            Ticket=int(Ticket)
            OpenPrice=float(OpenPrice)
            print("OPEN {} Ticket:{},OpenPrice:{}/{}".format(ORD_TYPE,Ticket,OpenPrice,err))
            return Ticket,OpenPrice, err

    def OrderClose(self,ticket=0, lots=0, price= 0):
            MSG = "OrderClose:"  + str(ticket) + '/lots:' + str(lots) + '/price:' + str(price)
            MSG = bytes(MSG+'\r\n', "ASCII")
            data ="11"
            self.cmnd.sendall(MSG)
            while True:
                data = self.cmnd.recv(4096).decode("ASCII")
                if  data: break            
            Ticket,ClosePrice,err = data.split(self.rx)
            Ticket=int(Ticket)
            ClosePrice=float(ClosePrice)
            print("CLOSE Ticket:{},ClosePrice:{}/{}".format(Ticket,ClosePrice,err))
            return Ticket,ClosePrice,err

    def AccountBalance(self):
            
            MSG = "AccountBalance"
            MSG = bytes(MSG+'\r\n', "ASCII")
            self.cmnd.sendall(MSG)
            data = self._rcmd()
            Balance = float(data)
            print("AccountBalance: {} ".format(Balance))
            return Balance
    def AccountEquity(self):
            
            MSG = "AccountEquity"
            MSG = bytes(MSG+'\r\n', "ASCII")            
            self.cmnd.sendall(MSG)
            #data=self.recv_end(self.cmnd)
            data = self._rcmd()
            Equity = float(data)
            print("AccountEquity: {} ".format(Equity))
            return Equity
    def AccountCurrency(self):
            
            MSG = "AccountCurrency"
            MSG = bytes(MSG+'\r\n', "ASCII")
            self.cmnd.sendall(MSG)
            data = self._rcmd()
            Currency = data
            print("AccountCurrency: {} ".format(Currency))
            return Currency






#mt4  = ServerMT4()
#time.sleep(15)
#mt4.stop()
#print('start')
##pd.show_versions()
if __name__== "__main__":
    mt4  = ClientMT4()
    BAL=mt4.AccountBalance()
    Eq=mt4.AccountEquity()
    Cr=mt4.AccountCurrency()
    print('Balance:',BAL,Eq,Cr)
    time.sleep(1.1)
    ##    #mt4.Subscribe_BidAsk(symbol='GBPUSD',last_ticks=100)
    ##    #mt4.bidask(symbol='GBPUSD')
    while 1:
    
        ticket,op, err = mt4.OrderSend(symbol='EURUSD', type= "OP_BUY", lots=0.01, price= 0, SL=0, TP=0, magic=12340,comment ='Python_to_MT')

        time.sleep(1.1)
        mt4.OrderClose(ticket=ticket, lots=0, price=0)
        ticket,op, err = mt4.OrderSend(symbol='EURUSD', type= "OP_SELL", lots=0.01, price= 0, SL=0, TP=0, magic=12340,comment ='Python_to_MT')
        time.sleep(1.1)
        mt4.OrderClose(ticket=ticket, lots=0, price=0)
#mt4.Subscribe_OHLC(symbol='EURUSD',timeframe=5,bars=50)

   

#import time
#import sys
#import win32pipe, win32file, pywintypes



## pipename should be of the form \\.\pipe\mypipename
#pipe = r'\\.\pipe\Pipe'
#MSG= b'AAAA'
#print("pipe server")
#BUFSIZE = 512
#p = win32pipe.CreateNamedPipe(pipe,
#                    win32pipe.PIPE_ACCESS_DUPLEX|
#                    win32file.FILE_FLAG_OVERLAPPED,
#                    win32pipe.PIPE_TYPE_MSG|
#                    win32pipe.PIPE_READMODE_MSG|
#                    win32pipe.PIPE_WAIT,
#                    1, BUFSIZE, BUFSIZE,
#                    win32pipe.NMPWAIT_USE_DEFAULT_WAIT,
#                    None)
#while True:
#    try:
#        print("waiting for client")
#        win32pipe.ConnectNamedPipe(p, None)
#        print("got client")        
#        resp = win32file.ReadFile(p,512)
#        #win32file.FlushFileBuffers(p)
#        win32pipe.DisconnectNamedPipe(p)
#        print(resp[1])
#        time.sleep(0.00001)
#        #win32file.CloseHandle(p)
#    except:
#        win32file.CloseHandle(p)
 

#import wpipe 

#pserver = wpipe.Server('Pipe', wpipe.Mode.Slave)
#print(pserver)
#while True:
#    for client in pserver:
#        while client.canread():
#            rawmsg = client.read()
#            print(rawmsg)
#            #client.write(b'hallo')    
#    pserver.waitfordata()
#pserver.shutdown()

        

#import time
#import struct


#f = open(r'\\.\pipe\Pipe',mode='r',newline="\n")
#i = 2

#while True:

#    s = 'MSG[{0}]'.format(i)
#    i += 1
#    f.tell()     
#    f.write(s)   # Write str length and str
#    f.seek(2)  
## EDIT: This is also necessary
#    print ('Wrote:', s)

#    #n = struct.unpack('I', f.read(4))[0]    # Read str length
#    #s = f.read(n)                           # Read str
#    #f.seek(0)                               # Important!!!
#    #print ('Read:', s)

#    time.sleep(1)