# %% [markdown]
# #  BM20A6100 Advanced Data Analysis and Machine Learning
# ## Erik Kuitunen, 0537275

# %% [markdown]
# ### Kernel Principal Component Analysis
# 

# %% [markdown]
# Read data, scale and visualize
# 

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Flowmeters/Meter B", delimiter="	", header=None)
col_names = df.columns.tolist() 

df

# %%
from sklearn.preprocessing import StandardScaler    

XData = df.iloc[ :, :-1 ].values
YData = df.iloc[ :, -1 ].values

scaler = StandardScaler()
scaler.fit( XData )
XData_scaled = scaler.transform( XData )
# YData_scaled = scaler.transform( YData.reshape(-1,1) )

pd.DataFrame( XData_scaled ).plot( kind = 'box', 
                                   figsize = (19,4), 
                                   rot = 90, 
                                   fontsize = 20, 
                                   sharey = False )


# %% [markdown]
# Defining kernels

# %%
def linear_kernel(X1, X2, sigma = None, degree = None):
    k_xixj = np.dot( X1, X2 )
    return k_xixj

def polynomial_kernel(X1, X2, sigma = None, degree = 2):
    k_xixj = ( np.dot( X1, X2 ) + 1 ) ** degree
    return k_xixj

def gaussian_kernel(X1, X2, sigma = 1, degree = None ):
    k_xixj = np.exp( -np.linalg.norm( X1 - X2 ) ** 2 / ( 2 * sigma ** 2 ) )
    return k_xixj
    

# %% [markdown]
# Calculating the Gram matrix using a kernel; centering 

# %%
# First using the linear kernel
def gram_centered(  kernel, X, Xhat = None, sigma = None, degree = None ):
    if kernel == 'linear':
        kernel = linear_kernel
    elif kernel == "polynomial":
        kernel = polynomial_kernel
    elif kernel == "gaussian":
        kernel = gaussian_kernel
        
    if Xhat is None:
        Nsamples_i = X.shape[0]
        Nsamples_j = X.shape[0]
        X = X.T
        # Xtest = X
    else:
        X = X.T
        Xhat = Xhat.T
        Nsamples_i = Xhat.shape[1]
        Nsamples_j = X.shape[1]      
        
    # First calculating K for calibration partition
    Kcal = np.zeros( (Nsamples_j, Nsamples_j) )    
    for ii in range(Nsamples_j):
        for jj in range(ii, Nsamples_j):
            elem = kernel( X[ :, ii ], X[ :, jj ], sigma = sigma, degree = degree )
            Kcal[ii, jj] = elem 
            if ii != jj:
                Kcal[jj, ii] = Kcal[ii, jj]
       
    # If fitting the model, return only the centered Gram matrix for calibration set
    if Xhat is None:
        I_n = np.eye( Nsamples_j )
        ones_n = np.ones((Nsamples_j, Nsamples_j))
        multiplier = ( I_n - ones_n / Nsamples_j @ ones_n.T )
        Kcal_hat = multiplier @ Kcal @ multiplier
        return Kcal_hat
    
    
    # If predicting, return the combined centered Gram matrix for calibration and prediction set
    K = np.zeros((Nsamples_i, Nsamples_j))
    for ii in range(Nsamples_i):
        for jj in range(Nsamples_j):
            elem = kernel( Xhat[ :, ii ], X[ :, jj ], sigma = sigma, degree = degree )
            K[ii, jj] = elem 

    ones_n = np.ones( (Nsamples_j, Nsamples_j) ) / Nsamples_j
    ones_ntest = np.ones( (Nsamples_i, Nsamples_j) ) / Nsamples_j
    K = K - ones_ntest @ Kcal - K @ ones_n + ones_ntest @ Kcal @ ones_n
    
    return K

# %% [markdown]
# Defining the kPLS algorithm

# %%
def kernel_PLS( K, y, n_components):
    n = K.shape[0]
    T = np.zeros( (n, n_components) )
    U = np.zeros( (y.shape[0], n_components) )
    
    print( "TU, 0: ", T.shape, U.shape )
    print( "y0:, ", y.shape )
    print( "K0:, ", K.shape )
    
    for ii in range( n_components ):
        
        t = K @ y
        t = t / np.linalg.norm(t)
        u = y @ ( y.T @ t )
        u = u / np.linalg.norm(u)
        
        print("KPLS:", t.shape, u.shape )
        
        # Store u and t
        T[ :, ii ] = t.reshape(-1)
        U[ :, ii ] = u.reshape(-1)
        
        # Deflate matrices
        K = K - t @ ( t.T @ K )
        y = y - t @ t.T
    
    return T, U

# %% [markdown]
# PLS function. No cross-validation, since amont of data is low.

# %%
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler    


def pls_model( X_train, y_train, X_test, y_test, n_comp, kernel = None, sigma = None, degree = None ):
    
    scalerX = StandardScaler().fit( X_train )
    X_train = scalerX.transform( X_train )
    X_test = scalerX.transform( X_test )
    
    scalerY = StandardScaler().fit( y_train.reshape(-1,1) )
    y_train = scalerY.transform( y_train.reshape(-1,1) )
    y_test = scalerY.transform( y_test.reshape(-1,1) )
    
    if kernel is None:
        model = PLSRegression( n_components = n_comp ).fit( X_train, y_train )
        y_pred_test = model.predict( X_test )
        
    else:
        K_train = gram_centered( kernel, X_train, sigma = sigma, degree = degree )
        K_test = gram_centered( kernel, X_train, Xhat = X_test, sigma = sigma, degree = degree )

        # For train data
        T, U = kernel_PLS( K_train, y_train, n_comp )
        B_train = U @ np.linalg.inv( T.T @ K_train @ U )  @ T.T @ y_train
        y_pred_train = K_train @ B_train
        
        # For test data
        T, U = kernel_PLS( K_test, y_test, n_comp )
        B_test = U @ np.linalg.inv( T.T @ K_test @ U )  @ T.T @ y_test
        y_pred_test = K_test @ B_test
        

    tss = np.sum( (y_train - np.mean(y_train)) ** 2 )
    press = np.sum( (y_pred_test - y_test) ** 2 )

    q2 = 1 - press / tss
    mse = root_mean_squared_error( y_test, y_pred_test )
    r2 = r2_score( y_train, y_pred_train )

    
    return y_pred_test, q2, r2, mse



# %% [markdown]
# Defining plotting function

# %%
def plot_results( q2vec, r2vec, mse_vec, Nvariables, kernel, y_test, y_test_pred ):
    components = np.arange(1, Nvariables+1)
    
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.plot( components, q2vec, '-', label = 'Q2' )
    plt.plot( components, r2vec, '-', label = 'R2')
    plt.title('Q2 and R2 vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('R2')
    plt.grid()
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot( components, mse_vec, marker='o' )
    plt.title( 'MSE vs Number of Components' )
    plt.xlabel( 'Number of Components' )
    plt.ylabel( 'MSE' )
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot( y_test, y_test_pred, 'o' )
    plt.plot( min(y_test), max(y_test), 'r--' )

    plt.suptitle(f"PLS metrics, kernel: {kernel}", )
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Test-train split, performing PLS and visualizing metrics

# %%
from sklearn.model_selection import train_test_split

def results():
    X_train, X_test, y_train, y_test = train_test_split( XData, YData, test_size = 0.2 )
    
    Nvariables = XData_scaled.shape[1]

    q2vec = np.zeros( Nvariables )
    r2vec = np.zeros( Nvariables )
    mse_vec = np.zeros( Nvariables )

    # for ii in range( Nvariables):
    #     y_pred, q2, r2, mse = pls_model( X_train, y_train, X_test, y_test, ii + 1, kernel = None )
    #     q2vec[ ii ] = q2
    #     r2vec[ ii ] = r2
    #     mse_vec[ ii ] = mse
        
    # plot_results( q2vec, r2vec, mse_vec, Nvariables, "No kernel", y_test, y_pred )

    for ii in range( Nvariables):
        y_pred, q2, r2, mse = pls_model( X_train, y_train, X_test, y_test, ii + 1, kernel = "linear" )
        q2vec[ ii ] = q2
        r2vec[ ii ] = r2
        mse_vec[ ii ] = mse
        
    plot_results( q2vec, r2vec, mse_vec, Nvariables, "Linear", y_test, y_pred )

    # for ii in range( Nvariables):
    #     y_pred, q2, r2, mse = pls_model( X_train, y_train, X_test, y_test, ii + 1, kernel = "gaussian", sigma = 1 )
    #     q2vec[ ii ] = q2
    #     r2vec[ ii ] = r2
    #     mse_vec[ ii ] = mse
        
    # plot_results( q2vec, r2vec, mse_vec, Nvariables, "Gaussian", y_test, y_pred )

    # for ii in range( Nvariables):
    #     y_pred, q2, r2, mse = pls_model( X_train, y_train, X_test, y_test, ii + 1, kernel = "polynomial", degree = 2 )
    #     q2vec[ ii ] = q2
    #     r2vec[ ii ] = r2
    #     mse_vec[ ii ] = mse
        
    # plot_results( q2vec, r2vec, mse_vec, Nvariables, "Polynomial", y_test, y_pred )

results()



