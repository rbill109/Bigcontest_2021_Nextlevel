import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import math
import re
import matplotlib.pyplot as plt
import seaborn as sns


class basemodel():


    def __init__(self,
                type = 'multiplicative',
                freq='1d',
                period=None):

        self.type = type
        if period == None:
            period = 365//int(re.sub(r'[^0-9]', '', freq))
            period = int(period)
        self.period = period
        self.freq = freq

    def make_df(self,df,emd_nm,col=None,col2=None,split=None,lag_col=None,x_lag=None,y_lag=None):

        self.col = col
        self.x_lag = x_lag
        self.y_lag = y_lag
        
        if lag_col == None:
            lag_col = self.col

        df = df[df['emd_nm'] == emd_nm]
        df.index = pd.DatetimeIndex(df.base_date)
        df["base_date"] = pd.to_datetime(df["base_date"])
        df = df.asfreq('1d',method='ffill')
        df = df.drop('base_date',axis=1)
        df = df.reset_index()
        df.index = pd.DatetimeIndex(df['base_date'])



        #freq기준에 따라 묶어주기
        if self.freq != '1d':
            if col2 == None:
                train_test = df.resample(self.freq,on='base_date',origin='end_day').sum()[['em_g',*self.col]][1:]
            else:
                df_sum = df.resample(self.freq,on='base_date',origin='end_day').sum()[set(['em_g',*self.col])-set(col2)][1:]
                df_mean = df.resample(self.freq,on='base_date',origin='end_day').mean()[col2][1:]
                train_test = train_test = pd.concat([df_sum,df_mean],axis=1)
            #train_test = train_test.asfreq(self.freq, method = 'ffill')
            train_test['emd_nm'] = emd_nm
        else:
            train_test = df


        # split 안하고 전체 데이터에 lag변수 추가
        if split == None:
            ind = 0
            new_col = []
            if x_lag != None:
                for i in lag_col:
                    train_test[i+'_shift'] = train_test[i].shift(x_lag)
                    new_col.append(i+'_shift')
                if x_lag>ind : ind = x_lag
                    
            if y_lag != None:
                train_test['em_g_shift'] = train_test['em_g'].shift(y_lag)
                new_col.append(i+'_shift')
                if y_lag>ind : ind = y_lag
            
            train_test = train_test.loc[train_test.index[ind:],:]
            self.new_col = [*new_col,*lag_col]
            return train_test

        else:
            train = train_test[:split]
            test = train_test[split:]

            #train data lag 변수
            ind = 0
            if x_lag != None:
                for i in lag_col:
                    train[i+'_shift'] = train[i].shift(x_lag)
                if x_lag>ind : ind = x_lag
                    
            if y_lag != None:
                train['em_g_shift'] = train['em_g'].shift(y_lag)
                if y_lag>ind : ind = y_lag
            
            train = train.loc[train.index[ind:],:]
            self.train = train

            #test data lag 변수

            self.na_y = 0
            new_col = []

            if x_lag != None:
                for i in lag_col:
                    shift = pd.DataFrame(train_test[i].shift(x_lag)) 
                    shift = shift.loc[shift.index.isin(test.index),:]
                    test[i+'_shift'] = shift
                    new_col.append(i+'_shift')

            if y_lag != None:
                shift = pd.DataFrame(train_test['em_g'].shift(self.y_lag))
                shift = shift.loc[shift.index.isin(test.index),:]
                if test.shape[0]-y_lag > 0:
                    self.na_y = test.shape[0]-y_lag
                    shift.loc[shift.index[-self.na_y:],'em_g'] = np.nan
                test['em_g_shift'] = shift
                new_col.append('em_g_shift')
            self.new_col = [*new_col,*lag_col]
            self.test = test
                
            return train,test

    

    def feature_importance(self,kind=None,show=False):
        try:
            dic = {'trend':self.model_trend,'seasonal':self.model_seasonal,'resid':self.model_resid}
        except:
            dic = {None:self.model}
        fig = plt.figure(figsize=(8,10))
        X_test = self.test[self.col]
        sns.barplot(y = X_test.columns[np.argsort(dic[kind].feature_importances_)], x = list(dic[kind].feature_importances_[np.argsort(dic[kind].feature_importances_)]))
        if show == False:
            plt.close()
        return fig


class basemodel_timeseries(basemodel):


    def predict(self,train,test,model):
        
        test['em_g_pred'] = np.nan

        pred = list()
        if (self.x_lag!=None) | (self.y_lag!=None):
            self.col = self.new_col
        
        if self.na_y != 0 :
            n = math.ceil(test.shape[0]/self.y_lag)
            for i in range(n):
                resid = test.shape[0]-(i+1)*self.y_lag
                model.fit(train[self.col],train['em_g'])
                if resid <= 0 :
                    n = test['em_g_pred'].isna().sum()
                    pred_block = list(model.predict(test[self.col][-n:]))
                    pred = pred + [pred_block]
                    test.loc[test.index[-n]:,'em_g_pred'] = pred_block
                else:
                    pred_block = list(model.predict(test[self.col][:self.train.shape[0]+(i+1)*self.y_lag]))
                    pred = pred + [pred_block]
                    test.loc[test.index[i*self.y_lag]:self.test.index[(i+1)*self.y_lag-1],'em_g_pred'] = pred_block
                    if resid >= self.y_lag:
                        test.loc[test.index[(i+1)*self.y_lag]:self.test.index[(i+2)*self.y_lag-1],'em_g_shift'] = pred_block
                    else:
                        test.loc[test.index[-resid]:,'em_g_shift'] = pred_block[:resid]

        else:
            model.fit(train[self.col],train['em_g'])
            pred_block = list(model.predict(test[self.col]))
            test.loc[:,'em_g_pred'] = pred_block
        
        self.model = model
        return test['em_g_pred'].values

class basemodel_timeseries_decomp(basemodel):


    def __init__(self,
                type = 'multiplicative',
                freq='1d',
                period=None):
        super().__init__(
                type = type,
                freq=freq,
                period=period)

    def decompose(self,df,col):
        
        df_trend = pd.DataFrame()
        df_seasonal = pd.DataFrame()
        df_resid = pd.DataFrame()
        
        for i in col:
            value = df[i]
            decomp = seasonal_decompose(value, model=self.type, extrapolate_trend='freq',period=self.period)
            df_resid[i] = decomp.resid
            df_seasonal[i] = decomp.seasonal
            df_trend[i] = decomp.trend
        
        return df_resid,df_seasonal,df_trend

    def combine(self,ar):
        if self.type == 'multiplicative':
            pred_comb = np.array(ar[0]) * np.array(ar[1]) * np.array(ar[2])
        else:
            pred_comb = np.array(ar[0]) + np.array(ar[1]) + np.array(ar[2])
        return pred_comb

    def get_model():
        return self.model_resid,self.model_seasonal,self.model_trend

    def predict(self,train,test,model,model2=None):
        
        test['em_g_pred'] = np.nan
        if model2 == None:
            model2 = model


        pred_decomp = list([[],[],[]])
        if (self.x_lag!=None) | (self.y_lag!=None):
            self.col = self.new_col
        
        #self.col = set(self.col) - set(cat)
        
        if self.na_y != 0 :
            n = math.ceil(test.shape[0]/self.y_lag)
            for i in range(n):
                resid = test.shape[0]-(i+1)*self.y_lag
                train_test = train.append(test) 

                pred = list()
                for j in range(3):
                    train_decomp = basemodel_timeseries_decomp.decompose(self,train,['em_g',*self.col])[j]
                    if j == 2:
                        model2.fit(train_decomp[self.col],train_decomp['em_g'])
                    else:
                        model.fit(train_decomp[self.col],train_decomp['em_g'])
                    if resid <= 0 :
                        n = test['em_g_pred'].isna().sum()
                        test_decomp = basemodel_timeseries_decomp.decompose(self,train_test,self.col)[j][-n:]
                    else:
                        test_decomp = basemodel_timeseries_decomp.decompose(self,train_test[:train.shape[0]+(i+1)*self.y_lag],self.col)[j][-self.y_lag:]

                    if j == 2:
                        pred_block = list(model2.predict(test_decomp[self.col])) #7 예측값, 분해된 상태
                    else:
                        pred_block = list(model.predict(test_decomp[self.col]))
                    pred = pred + [pred_block]
                    pred_decomp[j] = pred_decomp[j] + pred_block

                    if j==0 : self.model_resid = model
                    if j==1 : self.model_seasonal = model
                    if j==2 : self.model_trend = model2


                # em_g_shift, em_g 채워주기
                pred_comb = basemodel_timeseries_decomp.combine(self,pred)
                if resid > 0 : 
                    test.loc[test.index[i*self.y_lag]:test.index[(i+1)*self.y_lag-1],'em_g_pred'] = pred_comb
                    if resid >= self.y_lag:
                        test.loc[test.index[(i+1)*self.y_lag]:test.index[(i+2)*self.y_lag-1],'em_g_shift'] = pred_comb
                    else:
                        test.loc[test.index[-resid]:,'em_g_shift'] = pred_comb[:resid]
                elif resid <= 0:
                    n = test['em_g_pred'].isna().sum()
                    test.loc[test.index[-n]:,'em_g_pred'] = pred_comb[:n]
            

        else:
            pred = list()
            for i in range(3):
                train_test = train.append(test)
                train_decomp =  basemodel_timeseries_decomp.decompose(self,train,self.col)[i]
                test_decomp =  basemodel_timeseries_decomp.decompose(self,train_test,self.col)[i][-test.shape[0]:]
                train_y = basemodel_timeseries_decomp.decompose(self,train,['em_g'])[i]
                if i == 2:
                    model2.fit(train_decomp[self.col],train_y['em_g'])
                    pred_block = list(model2.predict(test_decomp[self.col]))
                    self.model_trend = model2
                else:
                    model.fit(train_decomp[self.col],train_y['em_g'])
                    if i==0 : self.model_resid = model
                    else: self.model_seasonal = model
                    pred_block = list(model.predict(test_decomp[self.col]))
                pred = pred + [pred_block]
                pred_decomp[i] = pred_decomp[i] + pred_block

                if i==0 : self.model_resid = model
                if i==1 : self.model_seasonal = model
                if i==2 : self.model_trend = model2

            pred_comb = basemodel_timeseries_decomp.combine(self,pred)
            test.loc[:,'em_g_pred'] = pred_comb


        return test['em_g_pred'].values,pred_decomp