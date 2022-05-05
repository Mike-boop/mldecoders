from scipy.linalg import toeplitz
from scipy.stats import pearsonr
from collections.abc import Iterable
import numpy as np

class Ridge:
    def __init__(self, start_lag=0, end_lag=50, alpha=1, verbose=True):
        '''
        num_lags: how many latencies to consider for the system response
        offset: when does the system response begin wrt an impulse timestamp? (may be negative)
        alpha: the regularisation parameter(s).
        '''
        self.start_lag=start_lag
        self.end_lag = end_lag
        self.num_lags = self.end_lag-self.start_lag
        if self.end_lag>0 and self.start_lag<0:
            self.num_lags+=1

        self.best_alpha_idx=False
        
        if isinstance(alpha, Iterable):
            self.alphas = alpha
        else:
            self.alphas = np.array([alpha])
            self.best_alpha_idx=0

        self.verbose = verbose

        
    def fit(self, X, y):
        '''
        inputs:
        - X, ndarray of shape (n_times, n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        '''
        
        # 1. Check that data shapes make sense

        if self.verbose:
            print("Checking inputs...")
        
        n_times, self.n_input_features = X.shape
        n_output_times, self.n_output_features = y.shape
        assert(n_times==n_output_times)
        
        # 2. Form the circulant data matrix

        if self.verbose:
            print("Formatting data matrix...")

        lagged_matrix = np.empty((n_times, self.num_lags, self.n_input_features))
        for ipf in range(self.n_input_features):
            lagged_matrix[:, :, ipf] = self._get_lagged_matrix(X[:, ipf])
            
        lagged_matrix = np.reshape(lagged_matrix, (n_times, self.num_lags*self.n_input_features))
        XtX = np.dot(lagged_matrix.T, lagged_matrix)
        
        # 3. Perform Ridge

        S, V = np.linalg.eigh(XtX)

        # Sort the eigenvalues
        s_ind = np.argsort(S)[::-1]
        S = S[s_ind]
        V = V[:, s_ind]

        # optional pcr stage

        # # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
        # # and compute the average
        # tol = np.finfo(float).eps
        # r = sum(S > tol)
        # #S = S[0:r]
        # #V = V[:, 0:r]
        # nl = np.mean(S)
        
        # 4. Apply ridge regression

        if self.verbose:
            print("Calculating coefficients...")

        self.coef_ = np.empty((self.alphas.size, self.n_output_features, self.n_input_features, self.num_lags))
        for i, alpha in enumerate(self.alphas):
            for j in range(self.n_output_features):

                XtY = np.dot(lagged_matrix.T, y[:, j])
                z = np.dot(V.T, XtY)
                tmp_coefs = V @ np.diag(1/(S+alpha)) @ z[:, np.newaxis]
                self.coef_[i, j, :, :] = np.reshape(tmp_coefs[:, 0], (self.num_lags, self.n_input_features)).T


    def predict(self, X, best_alpha=True):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - best_alpha: whether to make predictions for all regularisation parameters, or just the best one
        returns:
        - preditions, ndarray of shape (n_alphas, n_output_features, n_times)
        '''
        
        # 1. Form the Toeplitz matrix
        
        n_times = X.shape[0]
        lagged_matrix = np.empty((n_times, self.num_lags, self.n_input_features))
        
        for ipf in range(self.n_input_features):
            lagged_matrix[:, :, ipf] = self._get_lagged_matrix(X[:, ipf])
            
        lagged_matrix = np.reshape(lagged_matrix, (n_times, self.num_lags*self.n_input_features))
        
        # 2. Create predictions for every alpha and every output feature
                
        if best_alpha == False:
            
            predictions = np.empty((self.alphas.size, self.n_output_features, n_times))
            
            for i, alpha in enumerate(self.alphas):
                for j in range(self.n_output_features):
                    preds = lagged_matrix @ self.coef_[i, j].T.reshape(self.n_input_features*self.num_lags, 1)
                    predictions[i,j] = preds.flatten()
                    
        else:
            
            predictions = np.empty((self.n_output_features, n_times))
            for j in range(self.n_output_features):
                preds = lagged_matrix @ self.coef_[self.best_alpha_idx, j].T.reshape(self.num_lags*self.n_input_features, 1)
                predictions[j] = preds.flatten()

        return predictions
    
    
    def score(self, X, y, best_alpha=True, pad=False):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        - best_alpha: whether to score for best reg or all regularisation parameters
        - pad: whether to make predictions of the same size as y
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''
        
        predictions = self.predict(X, best_alpha=best_alpha)
        
        if best_alpha==False:
        
            scores = np.empty((self.alphas.size, self.n_output_features))
            for i, alpha in enumerate(self.alphas):
                for j in range(self.n_output_features):
                    scores[i, j] = pearsonr(predictions[i,j], y[:, j])[0]
        
        else:
            scores = np.empty((self.n_output_features))
            for j in range(self.n_output_features):
                scores[j] = pearsonr(predictions[j], y[:, j])[0]
            
        return scores

    
    def score_in_batches(self, X, y, batch_size=125):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''
        
        predictions = self.predict(X, best_alpha=True).T
        
        n_times = X.shape[0]
        num_batches = n_times // batch_size
        
        scores = []
        
        for batch_id in range(num_batches):
            p_batch = predictions[batch_id*batch_size:(batch_id+1)*batch_size]
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]
            scores.append([pearsonr(p_batch[:, opc], y_batch[:, opc])[0] for opc in range(self.n_output_features)])
            
        return np.asarray(scores)

    
    def model_selection(self, X, y):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - mean_scores, ndarray of shape (n_alphas,)
        
        also sets the attribute best_alpha_idx.
        '''
        
        scores = self.score(X, y, best_alpha=False)
        mean_scores = np.mean(scores, axis=1)
        self.best_alpha_idx = np.argmax(mean_scores)
        return mean_scores


    def _get_lagged_matrix(self, X):

        if self.start_lag<=0:
            X_ = np.pad(X, (abs(self.start_lag), 0))
            r = np.zeros(self.num_lags)
            lagged_matrix = toeplitz(X_, r)
            return lagged_matrix
        
        if self.start_lag>0:
            r = np.zeros((self.num_lags+self.start_lag))
            lagged_matrix = toeplitz(X, r)[:, self.start_lag:]
            return lagged_matrix