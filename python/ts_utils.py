import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_linear_trend_data(ts: np.ndarray, n_changepoints: int = 25, group_var=None):  
    """
    Parameters
    ----------
    ts : 1-D Array
        Time series.
    n_changepoints : int, default: 25
        Number of changepoints. 

    Returns
    -------
    tuple.
     t : Time index variable
     s : Changepoints
     A : Changepoint matrix
     
    References
    ----------
    
    .. [1] Taylor SJ, Letham B. 2017. Forecasting at scale. 
           PeerJ Preprints 5:e3190v2 https://doi.org/10.7287/peerj.preprints.3190v2

    """
    if group_var is None:
        n = len(ts)
        t = np.arange(n) / n
       
    else:
        groups ,counts = np.unique(group_var, return_counts=True)
        t = np.arange(counts[0]) / counts[0]
        t = np.hstack((t, ) * len(groups))
        
    s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    A = (t[:, None] > s) * 1. # t x s
          
    return t, s, A



def fourier_series(t: np.ndarray, p: float, fourier_order: int = 10):
    """
    Parameters
    ----------
    t : 1-D Array
        Time index.
    p : float
        Number of observations in one seasonality period.
    n_fourier : int
        Fourier order. Number of Fourier components to use.

    Returns
    -------
    2-D Array
        Seasonality matrix.
        
   
    References
    ----------
    
    .. [1] Taylor SJ, Letham B. 2017. Forecasting at scale. 
           PeerJ Preprints 5:e3190v2 https://doi.org/10.7287/peerj.preprints.3190v2

    """ 
    p = p / len(np.unique(t))
    X = 2 * np.pi * (np.arange(fourier_order) + 1) * t[:, None] / p
    X = np.concatenate((np.cos(X), np.sin(X)), axis=1)
    
    return X
    


def stan_model_summary(stan_fit):
    """
    Parameters
    ----------
    stan_fit : A Stan fit object

    Returns
    -------
    A pandas data frame with summary of posterior parameters. 

    """
    summary_dict = stan_fit.summary()
    summary_df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])
    return summary_df



def _forecast_train(stan_fit, p):
    """ Reproduce predictions on train data """
    
    train_n = stan_fit.data['N']
    t = np.arange(train_n) / train_n
    train_changepoints = stan_fit.data['s']
     
    A = (t[:, None] > train_changepoints) * 1. # t x s
    
    X = fourier_series(t, p=p, fourier_order=stan_fit.data['n_fourier']) 
    
    samples = stan_fit.extract(permuted=True)
    
    samples_n = len(samples['lp__'])
    
    y_hat = np.zeros(shape=(samples_n, len(t)))
    mu_trend = np.zeros(shape=(samples_n, len(t)))
    mu_s = np.zeros(shape=(samples_n, len(t)))
    
    for iteration in range(samples_n):
        
        k = samples['k'][iteration]
        m = samples['m'][iteration]
        deltas = samples['delta'][iteration]
        beta = samples['beta'][iteration]
        sigma = samples['sigma'][iteration]
        
        gamma = -train_changepoints * deltas;
        mu_trend[iteration,:] = k + np.matmul(A, deltas) * t + (m + np.matmul(A, gamma))
        mu_s[iteration,:] = np.matmul(X, beta)
        
        mu = mu_trend[iteration] + mu_s[iteration]
        
        y_hat[iteration,:] = np.random.normal(mu, sigma)
    
    return y_hat, mu_trend, mu_s, samples, samples_n, train_n, train_changepoints


def forecast(stan_fit, p = 4, horizon =10):
    """
    Parameters
    ----------
    stan_fit : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 4.
    horizon : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    y_hat : TYPE
        DESCRIPTION.
    mu_trend : TYPE
        DESCRIPTION.
    mu_s : TYPE
        DESCRIPTION.

    """
    
    y_hat_train, mu_trend_train, mu_s_train, samples, samples_n, train_n, train_changepoints = \
        _forecast_train(stan_fit, p = p) 
    
    t = np.arange(train_n+horizon)/train_n
    T = t.max()
    S = len(train_changepoints)
    
    p = p / train_n 
    X = 2 * np.pi * (np.arange(stan_fit.data['n_fourier']) + 1) * t[:, None] / p
    X = np.concatenate((np.cos(X), np.sin(X)), axis=1)
    
    y_hat = np.zeros(shape=(samples_n, len(t)))
    mu_trend = np.zeros(shape=(samples_n, len(t)))
    mu_s = np.zeros(shape=(samples_n, len(t)))
    
    for iteration in range(samples_n):
        
        n_changes = np.random.poisson(S * (T - 1))
        if n_changes > 0:
            changepoints_new = 1 + np.random.rand(n_changes) * (T - 1)
            changepoints_new.sort()
        else:
            changepoints_new = []
        changepoints = np.concatenate((train_changepoints,
                                       changepoints_new))
        
        k = samples['k'][iteration]
        m = samples['m'][iteration]
        deltas = samples['delta'][iteration]
        beta = samples['beta'][iteration]
        sigma = samples['sigma'][iteration]
    
        A = (t[:, None] > changepoints) * 1. # t x changepoints

        # Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
        lambda_ = np.mean(np.abs(deltas)) + 1e-8

        # Sample deltas
        deltas_new = np.random.laplace(0, lambda_, n_changes)
        deltas = np.concatenate((deltas, deltas_new))
        
        gamma = -changepoints * deltas;
        mu_trend[iteration,:] = k + np.matmul(A, deltas) * t + (m + np.matmul(A, gamma))
        mu_s[iteration,:] = np.matmul(X, beta)
        
        mu = mu_trend[iteration] + mu_s[iteration]
        
        y_hat[iteration,:] = np.random.normal(mu, sigma)
        
    # Replace train period with train predictions
    y_hat[:,0:train_n] = y_hat_train
    mu_trend[:,0:train_n] = mu_trend_train
    mu_s[:,0:train_n] = mu_s_train
    
    return y_hat, mu_trend, mu_s


def _forecast_hierarchical_train(stan_fit, p):

    J = stan_fit.data['J']
    train_n = int(stan_fit.data['N'] / J)
    t = np.arange(train_n) / train_n
    t = np.hstack((t, ) * stan_fit.data['J'])
    train_changepoints = stan_fit.data['s']
    A = (t[:, None] > train_changepoints) * 1. # t x s
    X = fourier_series(t, p=p, fourier_order=stan_fit.data['n_fourier']) 
    samples = stan_fit.extract(permuted=True)
    samples_n = len(samples['lp__'])
    group = stan_fit.data['group']    
   
        
    y_hat = np.zeros(shape=(samples_n, len(t)))
    mu_trend = np.zeros(shape=(samples_n, len(t)))
    mu_s = np.zeros(shape=(samples_n, len(t)))
       
    for iteration in range(samples_n):

        k = samples['k'][iteration]
        m = samples['m'][iteration]
        delta = samples['delta'][iteration]
        beta = samples['beta'][iteration]
        sigma = samples['sigma'][iteration]
        
        gamma = list()
        for j in range(J):
            gamma.append(-train_changepoints * delta[j])
            
        mu_s_ls = list()
        for j in range(1, J+1):
            mu_s_ls.append(np.matmul(X[group==j], beta[j-1])
                        )
        
        mu_trend_ls = list()
        for j in range(1, J+1):
            mu_trend_ls.append(
                k[j-1] + np.matmul(A[group==j], delta[j-1]) * t[group==j] + (m + np.matmul(A[group==j], gamma[j-1]))
                )
        mu_s[iteration,:] = np.concatenate(mu_s_ls, axis=0)
        mu_trend[iteration,:]  = np.concatenate(mu_trend_ls, axis=0)
        
        mu = mu_trend[iteration] + mu_s[iteration]
        y_hat[iteration,:] = np.random.normal(mu, sigma)
        
    return y_hat, mu_trend, mu_s, samples, samples_n, train_n, train_changepoints


def forecast_hierarchical(stan_fit, p=4, horizon=10):

    y_hat_train, mu_trend_train, mu_s_train, samples, samples_n, train_n, train_changepoints = \
        _forecast_hierarchical_train(stan_fit, p=p) 

    J = stan_fit.data['J']
    group = np.repeat(list(range(1,6)), (train_n+horizon))
    t = np.arange(train_n+horizon)/train_n
    t = np.hstack((t, ) * J)
    T = t.max()
    S = len(train_changepoints)

    p = p / train_n 
    X = 2 * np.pi * (np.arange(stan_fit.data['n_fourier']) + 1) * t[:, None] / p
    X = np.concatenate((np.cos(X), np.sin(X)), axis=1)

    y_hat = np.zeros(shape=(samples_n, len(t)))
    mu_trend = np.zeros(shape=(samples_n, len(t)))
    mu_s = np.zeros(shape=(samples_n, len(t)))
    
    for iteration in range(samples_n):
            
            n_changes = np.random.poisson(S * (T - 1))
            if n_changes > 0:
                changepoints_new = 1 + np.random.rand(n_changes) * (T - 1)
                changepoints_new.sort()
            else:
                changepoints_new = []
            changepoints = np.concatenate((train_changepoints,
                                           changepoints_new))
            
            k = samples['k'][iteration]
            m = samples['m'][iteration]
            delta = samples['delta'][iteration]
            beta = samples['beta'][iteration]
            sigma = samples['sigma'][iteration]
            
            A = (t[:, None] > changepoints) * 1. # t x changepoints
            
            deltas = np.empty(shape=(J, S+n_changes)) * np.NaN
            gammas = np.empty(shape=(J, S+n_changes)) * np.NaN
            mu_s_ls = list()
            mu_trend_ls = list()
            
            for j in range(J):    
                lambda_ = np.mean(np.abs(delta[j,:])) + 1e-8
                deltas_new = np.random.laplace(0, lambda_, n_changes)
                deltas[j,:] = (np.concatenate((delta[j,:], deltas_new)))
                gammas[j,:] = -changepoints * deltas[j,:]
                mu_s_ls.append(np.matmul(X[group==j+1], beta[j]))
                mu_trend_ls.append(
                    k[j] + np.matmul(A[group==j+1], deltas[j]) * t[group==j+1] + (m + np.matmul(A[group==j+1], gammas[j]))
                    )
            mu_s[iteration,:] = np.concatenate(mu_s_ls, axis=0)
            
            mu_trend[iteration,:]  = np.concatenate(mu_trend_ls, axis=0)
            
            mu = mu_trend[iteration] + mu_s[iteration]
            y_hat[iteration,:] = np.random.normal(mu, sigma)
        
    # Replace train period with train predictions
    replace_ind = t < 1. 
    y_hat[:,replace_ind] = y_hat_train
    mu_trend[:,replace_ind] = mu_trend_train
    mu_s[:,replace_ind] = mu_s_train
    
    return y_hat, mu_trend, mu_s


def plot_posterior(stan_fit, ts: np.array, ci: list = [0.05, 0.95],
                   yhat: np.array = None, horizon: int = None,
                   group = None, s: np.ndarray = None, 
                   component: str = None, title: str = None):
    """
    Parameters
    ----------
    stan_fit : A Stan fit object
    ts : A 1-D Array.
        Time series.
    ci: list of length 2.
       Endpoints of credibility interval.
    yhat: A 1-D Array.
        Predicted time series. 
    horizon: int
       Forecast horizon. If None, plots train data.
    group: str
        Grup indicator.
    s: 1-D Array
        Changepoints.
    component : str
        If `prediction`, plot show the posterior predictive 
        distribution of the response. If `trend`, it will
        show the trend component only. If `seasonality`,
        it will show the seasonality component only.
    title : str
        Plot title
    Returns
    -------
    Plot with posterior predictions (mean and credibility interval) vs actual values. 
    
    """
    
    n = stan_fit.data['N']  
    hierarchical = 'group' in stan_fit.data.keys()
   
    if hierarchical:
        group_train = np.unique(stan_fit.data['group'])
        n = n/len(group_train)
     
    if horizon is None:
    
        samples = stan_fit.extract(permuted=True)
        elem = samples.keys()
        t = np.arange(n) / n 
        if component is None:
            component = 'prediction'
       
        if component == 'prediction':
            if 'y_hat' in elem:
                r = samples['y_hat']
                if title is None:
                    title = 'Posterior Prediction'
            else:
                raise ValueError('Posterior prediction has not been \
                                 computed as part of the stan fit')
                  
        elif component == 'trend':
            if 'mu_trend' in elem:
                r = samples['mu_trend']
                if title is None:
                    title = 'Posterior Trend'
                    
            else:
                raise ValueError('Posterior trend has not been \
                                 computed as part of the stan fit')
                                 
        elif component == 'seasonality':
            if 'mu_s' in elem:
                r = samples['mu_s']
                if title is None:
                    title = 'Posterior Seasonality'
            else:
                raise ValueError('Posterior seasonality has not been \
                                 computed as part of the stan fit')
            
    else:
        if yhat is None: 
            raise ValueError('yhat must be provided when horizon = True')
        if component is not None:
            raise Warning("component=%s ignored when horizon=True" % component)
             
        t = np.arange(n+horizon) / n
        r = yhat
        vline = t[len(t) - horizon]
        
        if group is not None:
            if not hierarchical:
                raise Warning("Group argument provided but model fit is not hierarchical.")
            t = np.hstack((t, ) * len(group_train))
     
    r_mean = np.mean(r, axis=0)
    r_ci = np.quantile(r, ci, axis=0).T
    
    if hierarchical:
        df = pd.DataFrame({'y_hat': r_mean, 
                           't': t,
                           'y_hat_l': r_ci[:,0],
                           'y_hat_h': r_ci[:,1],
                           'group': group,
                           'y': ts})
        for g in np.unique(group):
            df_g = df.loc[df['group'] == g,:]
            fig, ax = plt.subplots(1)
            ax.plot(df_g['t'].values, df_g['y'].values, color='blue', marker='o')
            ax.plot(df_g['t'].values, df_g['y_hat'].values, color='red', marker='o')
            ax.fill_between(df_g['t'].values, df_g['y_hat_l'].values, df_g['y_hat_h'].values, facecolor='red', alpha=0.5)
            ax.set_title(g)
            ax.set_xlabel('Time')
            ax.set_ylabel('y')
            ax.axvline(x=vline, linestyle='dashed')
         
    else:
        
        fig, ax = plt.subplots(1)
        ax.plot(t, ts, color='blue', marker='o')
        ax.plot(t, r_mean, color='red', marker='o')
        ax.fill_between(t, r_ci[:,0], r_ci[:,1], facecolor='red', alpha=0.5)
        if s is not None:
            for sc in s:
                plt.axvline(x=sc, color='k', linestyle='--')
        if horizon is not None:
            ax.axvline(x=vline, linestyle='dashed')
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('y')
        ax.grid()
    
    return plt.show()

def plot_prophet(ts: np.array, ci: list = [0.05, 0.95],
                 yhat: np.array = None, horizon: int = None,
                 title: str = None):
    
    n = len(ts) - horizon 
    t = np.arange(n+horizon) / n
    vline = t[len(t) - horizon]
    r = yhat.T
      
    r_mean = np.mean(r, axis=0)
    r_ci = np.quantile(r, ci, axis=0).T
    
    fig, ax = plt.subplots(1)
    ax.plot(t, ts, color='blue', marker='o')
    ax.plot(t, r_mean, color='red', marker='o')
    ax.fill_between(t, r_ci[:,0], r_ci[:,1], facecolor='red', alpha=0.5)
    ax.axvline(x=vline, linestyle='dashed')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('y')
    ax.grid()
    
    return plt.show()
    
