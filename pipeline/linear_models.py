from scipy.linalg import toeplitz
from scipy.stats import pearsonr, spearmanr
from collections.abc import Iterable
import numpy as np

class Ridge:
    def __init__(self, num_lags, offset, alpha=1, scoring='pearson'):
        '''
        num_lags: how many latencies to consider for the system response
        offset: when does the system response begin wrt an impulse timestamp? (may be negative)
        '''
        self.num_lags = num_lags
        self.start = offset
        self.end = self.start+self.num_lags
        self.best_alpha_idx=False
        
        if isinstance(alpha, Iterable):
            self.alphas = alpha
        else:
            self.alphas = np.array([alpha])
            self.best_alpha_idx=0

        self.scoring = scoring

    def scorer(self, x, y):

        if self.scoring == 'pearson':
            return pearsonr(x,y)[1]
        if self.scoring == 'spearman':
            return spearmanr(x,y)[1]
        
        
    def fit(self, X, y, compress=True):
        '''
        inputs:
        - X, ndarray of shape (n_times, n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        '''
        
        # 1. Check that data shapes make sense

        print("Checking inputs...")
        
        n_times, self.n_input_features = X.shape
        n_output_times, self.n_output_features = y.shape
        assert(n_times==n_output_times)
        
        # 2. Form the circulant data matrix

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

        # Pick eigenvalues close to zero, remove them and corresponding eigenvectors
        # and compute the average
        tol = np.finfo(float).eps
        r = sum(S > tol)
        #S = S[0:r]
        #V = V[:, 0:r]
        nl = np.mean(S)


        
        # 4. Apply ridge regression

        print("Calculating coefficients...")
        self.coef_ = np.empty((self.alphas.size, self.n_output_features, self.n_input_features, self.num_lags))
        for i, alpha in enumerate(self.alphas):
            for j in range(self.n_output_features):

                XtY = np.dot(lagged_matrix.T, y[:, j])
                z = np.dot(V.T, XtY)
                tmp_coefs = V @ np.diag(1/(S+alpha)) @ z[:, np.newaxis]
                self.coef_[i, j, :, :] = np.reshape(tmp_coefs[:, 0], (self.num_lags, self.n_input_features)).T

        return None
                
    def predict(self, X, best_alpha=True):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
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
    
    
    def score(self, X, y, best_alpha=True, edge_correction=True):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''
        
        predictions = self.predict(X, best_alpha=best_alpha)
        
        if edge_correction==True:
            end_idx = -self.num_lags
        else:
            end_idx = None
        
        if best_alpha==False:
        
            scores = np.empty((self.alphas.size, self.n_output_features))
            for i, alpha in enumerate(self.alphas):
                for j in range(self.n_output_features):
                    scores[i, j] = self.scorer(predictions[i,j][:end_idx], y[:, j][:end_idx])
        
        else:
            scores = np.empty((self.n_output_features))
            for j in range(self.n_output_features):
                scores[j] = self.scorer(predictions[j][:end_idx], y[:, j][:end_idx])
            
        return scores
    
    def score_in_batches(self, X, y, batch_size=125, score='pearson', compression=True, compression_parameter=10):
        '''
        inputs:
        - X, ndarray of shape (n_times,  n_input_features)
        - y, ndarray of shape (n_times, n_output_features)
        returns:
        - scores, ndarray of shape (n_alphas, n_output_features)
        '''

        if score == 'pearson':
            scorer = pearsonr
        if score == 'spearman':
            scorer = spearmanr
        
        predictions = self.predict(X, best_alpha=True).T
        
        n_times = X.shape[0]
        num_batches = n_times // batch_size
        
        scores = []
        
        for batch_id in range(num_batches):
            p_batch = predictions[batch_id*batch_size:(batch_id+1)*batch_size]
            y_batch = y[batch_id*batch_size:(batch_id+1)*batch_size]

            if compression == True:
                y_ = np.tanh(compression_parameter*y_batch)
            else:
                y_ = y_batch
            scores.append([scorer(p_batch[:, opc], y_[:, opc])[0] for opc in range(self.n_output_features)])
            
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

        # positive lags
        if self.end > 0:
            r = np.zeros((self.end))
            positive_lags = toeplitz(X, r)
            if self.start > 0:
                return positive_lags[:, self.start:self.end]

            else:

                # negative lags
                r = np.zeros((-self.start + 1))
                negative_lags = toeplitz(X[::-1], r)[::-1, ::-1]

            # concat matrix
            matrix = np.hstack([negative_lags[:, :-1], positive_lags])
            
            return matrix

        r = np.zeros((-self.start))
        negative_lags = toeplitz(X[::-1], r)[::-1, ::-1]
        columns = self.end-self.start
        return negative_lags[:, :columns]