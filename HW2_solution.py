from pathlib import Path
import argparse
import json
import random
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE=Path(__file__).resolve().parent
OUT=BASE/'HW2_output'
OUT.mkdir(exist_ok=True)
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4,torch.get_num_threads()))


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT/name,dpi=160,bbox_inches='tight')
    plt.close()

def fit_early_stop(model, X_train, y_train, X_val, y_val, epochs=12, patience=3,
                   batch_size=1024, lr=0.002, loss_fn=None):
    if loss_fn is None: loss_fn=nn.MSELoss()
    train_x=torch.as_tensor(X_train)
    train_y=torch.as_tensor(y_train)
    val_x=torch.as_tensor(X_val)
    val_y=torch.as_tensor(y_val)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best=float('inf');best_state=None;wait=0;history=[]
    for ep in range(1,epochs+1):
        model.train();running=0.
        for batch_ids in torch.randperm(len(X_train)).split(batch_size):
            xb=train_x[batch_ids];yb=train_y[batch_ids]
            optimizer.zero_grad();yhat=model(xb)
            loss=loss_fn(yhat,yb)
            loss.backward();optimizer.step()
            running+=loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            v=loss_fn(model(val_x),val_y).item()
        history.append((ep,running/len(X_train),v))
        if v<best-0.0001:
            best=v; best_state={k:t.detach().clone() for k,t in model.state_dict().items()};wait=0
        else:
            wait+=1
        if wait>=patience:break
    model.load_state_dict(best_state)
    return history

def plot_history(history, name, title, ylabel='Loss'):
    a=np.array(history)
    plt.figure(figsize=(6,3.5));plt.plot(a[:,0],a[:,1],label='Training');plt.plot(a[:,0],a[:,2],label='Validation');plt.xlabel('Epoch');plt.ylabel(ylabel);plt.title(title);plt.legend();savefig(name)


class TaxiGaussianMLP(nn.Module):
    def __init__(self,p=7):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(p,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
        self.softplus=nn.Softplus()
    def forward(self,x):
        z=self.backbone(x)
        mu=z[:,0]
        sigma=self.softplus(z[:,1])+0.05
        return torch.stack((mu,sigma),dim=-1)

def gaussian_nll(out,y):
    mu=out[:,0];sigma=out[:,1]
    return (torch.log(sigma)+0.5*((y-mu)/sigma).square()+0.5*np.log(2*np.pi)).mean()

def taxi_predict(model,x,batch=16384):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch])).numpy() for i in range(0,len(x),batch)])

def gaussian_scores(mu,sigma,y):
    return dict(nll=float(np.mean(np.log(sigma)+0.5*((y-mu)/sigma)**2+0.5*np.log(2*np.pi))),
                rmse=float(np.sqrt(mean_squared_error(y,mu))))

def event_scores(mu,sigma,y):
    prob=np.clip(ndtr((45-mu)/sigma),1e-7,1-1e-7)
    true=(y<45).astype(int)
    return dict(brier=float(brier_score_loss(true,prob)),
                log_loss=float(log_loss(true,prob,labels=[0,1])),
                accuracy=float(np.mean((prob>=.5)==true)),
                event_rate=float(np.mean(true)),
                mean_probability=float(np.mean(prob)))

def run_problem1(results):
    print('\nRunning Problem 1...',flush=True)
    if (BASE/'_taxi_cache.npz').exists():
        a=np.load(BASE/'_taxi_cache.npz'); X=a['X'];y=a['y']
    else:
        try:
            dat=pd.read_parquet(BASE/'taxi_trips.parquet')
            names=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','min_of_day','day_of_week','day_of_year']
            X=dat[names].to_numpy(dtype=np.float32);y=dat['trip_duration'].to_numpy(dtype=np.float32)
        except ImportError:
            from _parquet_reader import load_parquet_columns
            names=['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','min_of_day','day_of_week','day_of_year']
            dat=load_parquet_columns(BASE/'taxi_trips.parquet',wanted=names+['trip_duration'])
            X=np.column_stack([dat[c] for c in names]).astype(np.float32);y=dat['trip_duration'].astype(np.float32)
    ix=np.arange(len(y))
    train_idx, temp_idx=train_test_split(ix,train_size=0.70,random_state=SEED)
    val_idx,test_idx=train_test_split(temp_idx,test_size=0.5,random_state=SEED)
    xtr=X[train_idx];xva=X[val_idx];xte=X[test_idx]
    ytr=y[train_idx];yva=y[val_idx];yte=y[test_idx]
    mu0=float(ytr.mean());sig0=float(ytr.std())
    lin=LinearRegression().fit(xtr,ytr)
    lmu_tr=lin.predict(xtr);lsig=float(np.sqrt(np.mean((ytr-lmu_tr)**2)))
    lin_mu=lin.predict(xte)
    base0=gaussian_scores(np.full_like(yte,mu0),np.full_like(yte,sig0),yte)
    base1=gaussian_scores(lin_mu,np.full_like(yte,lsig),yte)
    model=TaxiGaussianMLP()
    hist=fit_early_stop(model,xtr,ytr,xva,yva,epochs=14,patience=3,batch_size=1024,lr=.0015,loss_fn=gaussian_nll)
    p=taxi_predict(model,xte);m=p[:,0];s=p[:,1]
    nn_scores=gaussian_scores(m,s,yte)
    
    result={
        'n':len(y),'train_n':len(ytr),'val_n':len(yva),'test_n':len(yte),'training_epochs':len(hist),
        'constant_gaussian':{**base0,**event_scores(np.full_like(yte,mu0),np.full_like(yte,sig0),yte)},
        'linear_gaussian':{**base1,**event_scores(lin_mu,np.full_like(yte,lsig),yte)},
        'heteroscedastic_mlp':{**nn_scores,**event_scores(m,s,yte)}
    }
    plot_history(hist,'p1_training.png','Taxi Gaussian MLP training','Gaussian NLL')
    prob=ndtr((45-m)/s);truth=(yte<45).astype(float)
    breaks=np.quantile(prob,np.linspace(0,1,7));breaks=np.unique(breaks)
    if len(breaks)>2:
        inds=np.digitize(prob,breaks[1:-1])
        q=[];obs=[]
        for k in range(len(breaks)-1):
            mask=inds==k
            if np.any(mask):q.append(prob[mask].mean());obs.append(truth[mask].mean())
        plt.figure(figsize=(5,4));plt.plot(q,obs,'o-',label='Gaussian MLP');plt.plot([.93,1],[.93,1],'--',label='Ideal')
        plt.xlabel('Mean predicted P(duration < 45)');plt.ylabel('Observed fraction');plt.title('Taxi event calibration');plt.legend();savefig('p1_calibration.png')
    print(json.dumps(result,indent=2),flush=True)
    results['problem1']=result

class SmoothedTargetEncoder:
    """Training-split-only category mean on log1p(Quantity), regularized for rare levels."""
    def __init__(self,smoothing=20):
        self.smoothing=smoothing
    def fit(self,frame,target):
        self.cols=['StockCode','CustomerID']
        self.global_mean=float(np.mean(target))
        self.maps=[]
        for col in self.cols:
            z=pd.DataFrame({'cat':frame[col].values,'target':target})
            s=z.groupby('cat',sort=False)['target'].agg(['mean','count'])
            enc=(s['count']*s['mean']+self.smoothing*self.global_mean)/(s['count']+self.smoothing)
            self.maps.append(enc)
        return self
    def transform(self,frame):
        return np.column_stack([frame[c].map(t).fillna(self.global_mean).to_numpy(dtype=np.float32) for c,t in zip(self.cols,self.maps)])

def crossfit_target_encode(frame,target,n_folds=5,smoothing=20):
    out=np.empty((len(frame),2),dtype=np.float32)
    kf=KFold(n_splits=n_folds,shuffle=True,random_state=SEED)
    for tr,va in kf.split(frame):
        enc=SmoothedTargetEncoder(smoothing).fit(frame.iloc[tr],target[tr]);out[va]=enc.transform(frame.iloc[va])
    full=SmoothedTargetEncoder(smoothing).fit(frame,target)
    return out,full

def id_lookup(tr,va,te,col):
    known=pd.Index(tr[col].unique());keys={s:i+1 for i,s in enumerate(known)}
    def enc(data):return data[col].map(keys).fillna(0).to_numpy(dtype=np.int64)
    return enc(tr),enc(va),enc(te),len(known)+1

class RetailEmbMLP(nn.Module):
    def __init__(self,nstock,ncust,combined=False):
        super().__init__()
        self.embstock=nn.Embedding(nstock,16)
        self.embcust=nn.Embedding(ncust,16)
        self.combined=combined
        self.mlp=nn.Sequential(nn.Linear(32+(2 if combined else 0),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        a=self.embstock(x[:,0].long());b=self.embcust(x[:,1].long())
        z=torch.cat((a,b,x[:,2:].float()),axis=1) if self.combined else torch.cat((a,b),axis=1)
        return self.mlp(z).flatten()

def retail_eval(y_log,pred_log):
    raw=np.expm1(np.clip(pred_log,0,13))
    return dict(log_mse=float(mean_squared_error(y_log,pred_log)),
                log_rmse=float(np.sqrt(mean_squared_error(y_log,pred_log))),
                quantity_mae=float(mean_absolute_error(np.expm1(y_log),raw)))

def run_problem2(results):
    print('\nRunning Problem 2...',flush=True)
    dat=pd.read_csv(BASE/'retail.csv',dtype={'StockCode':'string','CustomerID':'string'})
    dat['StockCode']=dat['StockCode'].astype(str);dat['CustomerID']=dat['CustomerID'].astype(str)
    ix=np.arange(len(dat))
    train_idx,temp_idx=train_test_split(ix,train_size=.80,random_state=SEED)
    val_idx,test_idx=train_test_split(temp_idx,test_size=.50,random_state=SEED)
    tr=dat.iloc[train_idx].reset_index(drop=True)
    va=dat.iloc[val_idx].reset_index(drop=True)
    te=dat.iloc[test_idx].reset_index(drop=True)
    target=np.log1p(tr['Quantity'].to_numpy(dtype=np.float32))
    yva=np.log1p(va['Quantity'].to_numpy(dtype=np.float32))
    yte=np.log1p(te['Quantity'].to_numpy(dtype=np.float32))
    enc_tr,enc=crossfit_target_encode(tr,target)
    enc_va=enc.transform(va);enc_te=enc.transform(te)
    target_model=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=0.07,
                                                  min_samples_leaf=30,l2_regularization=0.1,random_state=SEED)
    target_model.fit(enc_tr,target)
    te_pred=target_model.predict(enc_te)
    res={'n':len(dat),'train_n':len(tr),'val_n':len(va),'test_n':len(te),
         'stock_levels':int(dat.StockCode.nunique()),'customer_levels':int(dat.CustomerID.nunique()),
         'quantity_max':int(dat.Quantity.max()),
         'target_encoding':retail_eval(yte,te_pred),
         'constant_baseline':retail_eval(yte,np.full_like(yte,target.mean()))}
    stock_tr,stock_va,stock_te,nstock=id_lookup(tr,va,te,'StockCode')
    cust_tr,cust_va,cust_te,ncust=id_lookup(tr,va,te,'CustomerID')
    xtr=np.column_stack([stock_tr,cust_tr]).astype(np.float32)
    xva=np.column_stack([stock_va,cust_va]).astype(np.float32)
    xte=np.column_stack([stock_te,cust_te]).astype(np.float32)
    scaler_mu=enc_tr.mean(axis=0);scaler_sd=np.maximum(enc_tr.std(axis=0),1e-5)
    xtr_full=np.column_stack((xtr,(enc_tr-scaler_mu)/scaler_sd)).astype(np.float32)
    xva_full=np.column_stack((xva,(enc_va-scaler_mu)/scaler_sd)).astype(np.float32)
    xte_full=np.column_stack((xte,(enc_te-scaler_mu)/scaler_sd)).astype(np.float32)
    for combined,key in [(False,'embedding'),(True,'embedding_plus_target')]:
        torch.manual_seed(SEED+int(combined))
        model=RetailEmbMLP(nstock,ncust,combined)
        xt,xv,xh=(xtr_full,xva_full,xte_full) if combined else (xtr,xva,xte)
        hist=fit_early_stop(model,xt,target,xv,yva,epochs=8,patience=2,batch_size=4096,lr=.004)
        model.eval()
        with torch.no_grad():
            preds=np.concatenate([model(torch.tensor(xh[i:i+32768])).numpy() for i in range(0,len(xh),32768)])
        res[key]={**retail_eval(yte,preds),'training_epochs':len(hist),'validation_log_mse':float(min(row[2] for row in hist))}
        plot_history(hist,'p2_'+key+'.png','Retail: '+key.replace('_',' '),'Log-scale MSE')
    results['problem2']=res
    print(json.dumps(res,indent=2),flush=True)

class RKS:
    def __init__(self,D=160,sigma=1.,lam=0.01,seed=SEED):
        assert D%2==0
        self.D=D;self.sigma=sigma;self.lam=lam;self.seed=seed
    def _features(self,X):
        h=self.D//2
        angles=np.asarray(X)@self.W+self.b
        return np.column_stack([np.ones(len(X)),np.sin(angles),np.cos(angles)])
    def fit(self,X,y):
        rng=np.random.default_rng(self.seed)
        self.W=rng.normal(size=(X.shape[1],self.D//2))/self.sigma
        self.b=rng.uniform(0,2*np.pi,size=(self.D//2,))
        P=self._features(X)
        A=P.T@P;A.flat[::len(A)+1]+=self.lam
        A[0,0]-=self.lam
        self.coef=solve(A,P.T@y,assume_a='pos')
        return self
    def predict(self,X):return self._features(X)@self.coef

class ToyMLP(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,hidden),nn.Tanh(),nn.Linear(hidden,1))
    def forward(self,x):return self.net(x).reshape(-1)

def train_toy_mlp(hidden,X,y,epochs=450,seed=SEED):
    torch.manual_seed(seed)
    model=ToyMLP(hidden)
    opt=torch.optim.Adam(model.parameters(),lr=.008)
    inp=torch.from_numpy(X).float();tgt=torch.from_numpy(y).float()
    for ep in range(epochs):
        model.train();opt.zero_grad();pred=model(inp)
        loss=torch.mean((pred-tgt)**2);loss.backward();opt.step()
    model.eval();return model

def toy_predict(model,x):
    with torch.no_grad():return model(torch.from_numpy(x).float()).numpy()

def plot_sweep(param,train,val,xlabel,fileprefix):
    for v,name in [(train,'Training'),(val,'Validation')]:
        plt.figure(figsize=(5.8,3.5));plt.plot(param,v,'o-',color='#4169ad');plt.xlabel(xlabel);plt.ylabel('Mean squared error');plt.title(f'{fileprefix}: {name.lower()} MSE');plt.grid(alpha=.25)
        savefig(f'p3_{fileprefix.lower()}_{name.lower()}.png')

def run_problem3(results):
    print('\nRunning Problem 3...',flush=True)
    tr=pd.read_csv(BASE/'train.csv');va=pd.read_csv(BASE/'val.csv');te=pd.read_csv(BASE/'test.csv')
    X=tr[['x1','x2']].to_numpy();y=tr.y.to_numpy()
    V=va[['x1','x2']].to_numpy();vy=va.y.to_numpy()
    T=te[['x1','x2']].to_numpy()
    plt.figure(figsize=(6,5));plt.scatter(X[:,0],X[:,1],c=y,s=27,cmap='viridis',edgecolor='none');plt.colorbar(label='y');plt.xlabel('x1');plt.ylabel('x2');plt.xlim(-3,3);plt.ylim(-3,3);plt.title('Training samples');savefig('p3_training_data.png')
    ds=[20,40,80,160,320,640]
    rks_train=[];rks_val=[]
    for D in ds:
        m=RKS(D=D).fit(X,y)
        rks_train.append(mean_squared_error(y,m.predict(X)))
        rks_val.append(mean_squared_error(vy,m.predict(V)))
    bestD=ds[int(np.argmin(rks_val))]
    plot_sweep(ds,rks_train,rks_val,'Random features (D)','RKS')
    hs=[8,16,32,64,128]
    mlp_train=[];mlp_val=[]
    for h in hs:
        model=train_toy_mlp(h,X,y)
        mlp_train.append(mean_squared_error(y,toy_predict(model,X)))
        mlp_val.append(mean_squared_error(vy,toy_predict(model,V)))
    bestH=hs[int(np.argmin(mlp_val))]
    plot_sweep(hs,mlp_train,mlp_val,'Hidden units','MLP')
    rks=RKS(D=bestD).fit(X,y);mlp=train_toy_mlp(bestH,X,y)
    rpred=rks.predict(T);mpred=toy_predict(mlp,T)
    np.savetxt(BASE/'rks_test_predictions.csv',rpred,delimiter=',',fmt='%.10f')
    np.savetxt(BASE/'mlp_test_predictions.csv',mpred,delimiter=',',fmt='%.10f')
    g=np.linspace(-3,3,160);xx,yy=np.meshgrid(g,g);grid=np.c_[xx.ravel(),yy.ravel()]
    for model,name in [(rks,'RKS'),(mlp,'MLP')]:
        z=model.predict(grid) if name=='RKS' else toy_predict(model,grid)
        fig=plt.figure(figsize=(6,5))
        plt.contourf(xx,yy,z.reshape(xx.shape),levels=30,cmap='viridis');plt.colorbar(label='Predicted y');plt.xlabel('x1');plt.ylabel('x2');plt.title(name+' fitted surface');savefig('p3_'+name.lower()+'_surface.png')
    res={'train_n':len(tr),'val_n':len(va),'test_n':len(te),
         'rks':{'hyperparameter':'D','sweep':ds,'train_mse':[float(x) for x in rks_train],
                'val_mse':[float(x) for x in rks_val],'best_D':bestD,'best_val_mse':float(min(rks_val))},
         'mlp':{'hyperparameter':'hidden units','sweep':hs,'train_mse':[float(x) for x in mlp_train],
                'val_mse':[float(x) for x in mlp_val],'best_hidden':bestH,'best_val_mse':float(min(mlp_val))}}
    results['problem3']=res
    print(json.dumps(res,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',type=int,choices=[1,2,3],help='Run one part (default all)')
    args=parser.parse_args()
    p=BASE/'HW2_output'/'metrics.json'
    results=json.loads(p.read_text()) if p.exists() else {}
    for i,fn in [(1,run_problem1),(2,run_problem2),(3,run_problem3)]:
        if args.part is None or args.part==i:
            fn(results);p.write_text(json.dumps(results,indent=2))
    print('Wrote results to',p,flush=True)

if __name__=='__main__':main()
