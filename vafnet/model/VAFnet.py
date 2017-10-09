# Copyright 2017 Max W. Y. Lam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys, os, string, time
import numpy as np
import numpy.random as npr

import theano as T
from theano import shared as Ts, function as Tf, config as Tc, tensor as Tt
from theano.sandbox import linalg as Tlin

from .. import Scaler, Metric, Optimizer, Visualizer
    
"""
VAFnet: Variational Activation Functions for Deep Learning
"""

__all__ = [
    'VAFnet',
]


DEBUG = False
def debug(str, var):
    if(DEBUG):
        return T.printing.Print(str)(var)
    return var


class VAFnet(object):
    
    '''
    The :class:`Model` class implemented handy functions shared by all machine
    learning models. It is always called as a subclass for any new model.
    
    Parameters
    ----------
    X_trans : a string
        Transformation method used for inputs of training data
    Y_trans : a string
        Transformation method used for outpus of training data
    verbose : a bool
        Idicator that determines whether printing training message or not
    '''

    verbose = True
    X, Y, X_Trans, Y_Trans, params, compiled_funcs, trained_mats = [None]*7
    
    def __init__(self, archit=[50], verbose=True, **args):
        Xt = 'normal' if 'X_trans' not in args.keys() else args['X_trans']
        Yt = 'normal' if 'Y_trans' not in args.keys() else args['Y_trans']
        self.scaler = {'X': Scaler(Xt), 'Y': Scaler(Yt)}
        self.verbose = verbose
        self.archit = archit
        self.D = [-1]+(2*np.array(archit)).tolist()+[-1]
        self.H = len(self.archit)
        self.E_meth = 'MC' if 'E_meth' not in args.keys() else args['E_meth']
        self.float_type = Tc.floatX
        rand_str = ''.join(chr(npr.choice([ord(c) for c in (
            string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.ID = self.__str__()+'-'+rand_str
        self.evals = {
            'score': ['Model Selection Score', []],
            'obj': ['Objective Function Value', []],
            'mse': ['Mean Square Error', []],
            'rmse': ['Mean Square Error', []],
            'nmse': ['Normalized Mean Square Error', []],
            'mnlp': ['Mean Negative Log Probability', []],
            'time': ['Training Time (second)', []],
        }
    
    def __str__(self):
        return "VAFnet("+",".join([str(D_h) for D_h in self.archit])+")"
    
    def echo(self, *arg):
        if(self.verbose):
            print(' '.join(map(str, arg)))
            sys.stdout.flush()
    
    def get_params(self, **arg):
        if(self.params is None):
            self.echo('-'*80, '\nInitializing hyperparameters...')
            self.params = Ts(self.randomized_params().astype(self.float_type))
            self.echo('done.')
        return self.params.eval()
    
    def get_compiled_funcs(self):
        return self.compiled_funcs
    
    def unpack_trained_mats(self, trained_mats):
        return {'obj': np.double(trained_mats[0]),
                'alpha': trained_mats[1],
                'Linv': trained_mats[2],
                'mu_F': trained_mats[3]}
    
    def unpack_predicted_mats(self, predicted_mats):
        return {'mu_Fs': predicted_mats[0],
                'std_Fs': predicted_mats[1],}
    
    def pack_train_func_inputs(self, X, Y):
        return [X.astype(self.float_type), Y.astype(self.float_type)]
    
    def pack_pred_func_inputs(self, Xs):
        return [Xs.astype(self.float_type),
            self.trained_mats['alpha'].astype(self.float_type),
            self.trained_mats['Linv'].astype(self.float_type)]

    def fit(self, X, Y, update_params=False):
        X, Y = X.astype(self.float_type), Y.astype(self.float_type)
        self.Xt = self.scaler['X'].transform(X)
        self.Yt = self.scaler['Y'].transform(Y)
        (self.D[0], self.N), self.D[-1] = self.Xt.shape, self.Yt.shape[0]
        self.ms_len = np.sum([self.D[d]*self.D[d+1] for d in range(self.H)])
        self.Ab_len = np.sum([self.D[d] for d in range(self.H)])
        if(self.params is None):
            self.echo('-'*80, '\nInitializing hyperparameters...')
            self.params = Ts(self.randomized_params())
            self.echo('done.')
        else:
            trained_mats = self.compiled_funcs['opt' if update_params else
                'train'](self.Xt, self.Yt)
            self.trained_mats = self.unpack_trained_mats(trained_mats)

    def randomized_params(self):
        rand_params = npr.randn(1+int(self.ms_len+self.Ab_len))
        return rand_params.astype(self.float_type)
    
    def feature_maps(self, X, params):
        pid = 0
        lg_sig2 = params[pid:pid+1]; pid+=1
        F, M, V = [], [], []
        M_shape = (self.D[1]//2, self.D[0])
        m = params[pid:pid+np.prod(M_shape)]; pid+=np.prod(M_shape)
        M.append(Tt.reshape(m, M_shape))
        V_shape = (self.D[1]//2, self.D[0])
        s = params[pid:pid+np.prod(V_shape)]; pid+=np.prod(V_shape)
        V.append(Tt.reshape(s**2, V_shape)/np.prod(V_shape))
        b = params[pid:pid+self.D[0]]; pid+=self.D[0]
        b = Tt.reshape(b, (self.D[0], 1))
        Z = X+b
        MZ = debug('MZ', 2*np.pi*Tt.dot(M[-1], Z))
        expVZ = debug('expVZ', Tt.exp(-2*np.pi**2*Tt.dot(V[-1], Z**2)))
        F.append(Tt.concatenate([expVZ*Tt.cos(MZ), expVZ*Tt.sin(MZ)]))
        for h in range(self.H-1):
            M_shape = (self.D[h+2]//2, self.D[h+1])
            m = params[pid:pid+np.prod(M_shape)]; pid+=np.prod(M_shape)
            M.append(Tt.reshape(m, M_shape))
            V_shape = (self.D[h+2]//2, self.D[h+1])
            s = params[pid:pid+np.prod(V_shape)]; pid+=np.prod(V_shape)
            V.append(Tt.reshape(s**2, V_shape)/np.prod(V_shape))
            b = params[pid:pid+self.D[h+1]]; pid+=self.D[h+1]
            b = Tt.reshape(b, (self.D[h+1], 1))
            Z = F[-1]+b
            MZ = 2*np.pi*Tt.dot(M[-1], Z)
            expVZ = Tt.exp(-2*np.pi**2*Tt.dot(V[-1], Z**2))
            F.append(Tt.concatenate([expVZ*Tt.cos(MZ), expVZ*Tt.sin(MZ)]))
        if(type(X) == Tt.TensorVariable):
            return Tt.exp(lg_sig2)[0], F, M, V
        return Phi

    def compile_theano_funcs(self, opt_algo, opt_params, dropout):
        self.compiled_funcs = {}
        # Compile Train & Optimization Function
        eps = 1e-5
        params = Tt.vector('params')
        X, Y = Tt.matrix('X'), Tt.matrix('Y')
        sig2, F, M, V = self.feature_maps(X, params)
        EPhi = F[-1]
        EPhiPhiT = Tt.dot(EPhi, Tt.transpose(EPhi))
        A = EPhiPhiT+(sig2+eps)*Tt.identity_like(EPhiPhiT)
        L = Tlin.cholesky(A)
        Linv = Tlin.matrix_inverse(L)
        YPhiT = Tt.dot(Y, Tt.transpose(EPhi))
        beta = Tt.dot(YPhiT, Tt.transpose(Linv))
        alpha = Tt.dot(beta, Linv)
        mu_F = Tt.dot(alpha, EPhi)
        GOF = .5/sig2*Tt.sum(Tt.sum(Tt.dot(Y,(Y-mu_F).T)))
        REG = Tt.sum(Tt.log(Tt.diagonal(L)))+(self.N-self.D[-2])/2*Tt.log(sig2)
        REG *= self.D[-1]
        KL = 0
        for h in range(self.H):
            KL += Tt.sum(Tt.sum(M[h]**2)+Tt.sum(V[h]-Tt.log(V[h]+eps)))
            KL -= self.D[h+1]*self.D[h+2]//2
        obj = debug('obj', GOF+REG+KL)
        self.compiled_funcs['debug'] = Tf([X, Y], [obj],
            givens=[(params, self.params)])
        grads = Tt.grad(obj, params)
        updates = {self.params: grads}
        updates = getattr(Optimizer, opt_algo)(updates, **opt_params)
        updates = getattr(Optimizer, 'nesterov')(updates, momentum=0.9)
        train_inputs = [X, Y]
        train_outputs = [obj, alpha, Linv, mu_F]
        self.compiled_funcs['opt'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)], updates=updates)
        self.compiled_funcs['train'] = Tf(train_inputs, train_outputs,
            givens=[(params, self.params)])
        # Compile Predict Function
        Linv, alpha = Tt.matrix('Linv'), Tt.matrix('alpha')
        Xs = Tt.matrix('Xs')
        sig2, Fs, _, _ = self.feature_maps(Xs, params)
        EPhis = Fs[-1]
        mu_Fs = Tt.dot(alpha, EPhis)
        std_Fs = ((sig2*(1+(Tt.dot(Linv, EPhis)**2).sum(0)))**0.5)[:, None]
        pred_inputs = [Xs, alpha, Linv]
        pred_outputs = [mu_Fs, std_Fs]
        self.compiled_funcs['pred'] = Tf(pred_inputs, pred_outputs,
            givens=[(params, self.params)])

    def score(self, X, Y):
        self.Xs = self.scaler['X'].transform(X)
        self.Ys = self.scaler['Y'].transform(Y)
        mu, std = self.predict(X)
        mse = np.mean((mu-Y)**2.)
        self.evals['mse'][1].append(mse)
        rmse = np.sqrt(mse)
        self.evals['rmse'][1].append(rmse)
        nmse = mse/np.var(Y)
        self.evals['nmse'][1].append(nmse)
        mnlp = 0.5*np.mean(((Y-mu)/std)**2+np.log(2*np.pi*std**2))
        self.evals['mnlp'][1].append(mnlp)
        score = nmse/(1+np.exp(-mnlp))
        self.evals['score'][1].append(score)
        return score

    def cross_validate(self, X, Y, nfolds, change=True, iter=0, callback=None):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=nfolds, random_state=None)
        cv_evals_sum = {metric: [] for metric in self.evals.keys()}
        for train, valid in kf.split(X.T):
            Xt, Yt = X[:,train], Y[:,train]
            Xv, Yv = X[:,valid], Y[:,valid]
            self.fit(Xt, Yt, change)
            cv_evals_sum['obj'].append(self.trained_mats['obj'])
            self.score(Xv, Yv)
            if(callback is not None):
                callback(iter)
            for metric in self.evals.keys():
                if(metric == 'obj' or metric == 'time'):
                    continue
                cv_evals_sum[metric].append(self.evals[metric][1].pop())
        self.fit(X, Y, change)
        cv_evals_sum['time'].append(time.time()-self.train_start_time)
        cv_evals_sum['obj'].append(self.trained_mats['obj'])
        for metric in self.evals.keys():
            self.evals[metric][1].append(np.mean(cv_evals_sum[metric]))

    def optimize(self, X, Y, funcs=None, visualizer=None, **args):
        X, Y = X.astype(self.float_type), Y.astype(self.float_type)
        self.scaler['X'].fit(X); self.scaler['Y'].fit(Y)
        self.fit(X, Y)
        obj_type = 'obj' if 'obj' not in args.keys() else args['obj'].lower()
        obj_type = 'obj' if obj_type not in self.evals.keys() else obj_type
        opt_algo = {'algo': None} if 'algo' not in args.keys() else args['algo']
        cv_nfolds = 5 if 'cv_nfolds' not in args.keys() else args['cv_nfolds']
        cvrg_tol = 1e-4 if 'cvrg_tol' not in args.keys() else args['cvrg_tol']
        max_cvrg = 18 if 'max_cvrg' not in args.keys() else args['max_cvrg']
        max_iter = 500 if 'max_iter' not in args.keys() else args['max_iter']
        dropout = 1. if 'dropout' not in args.keys() else args['dropout']
        if(opt_algo['algo'] not in Optimizer.algos):
            opt_algo = {
                'algo': 'adam',
                'algo_params': {
                    'learning_rate':0.01,
                    'beta1':0.9,
                    'beta2':0.999,
                    'epsilon':1e-8
                }
            }
        for metric in self.evals.keys():
            self.evals[metric][1] = []
        if(funcs is None):
            self.echo('-'*80, '\nCompiling theano functions...')
            algo, algo_params = opt_algo['algo'], opt_algo['algo_params']
            self.compile_theano_funcs(algo, algo_params, dropout)
            self.echo('done.')
        else:
            self.compiled_funcs = funcs
        if(visualizer is not None):
            visualizer.model = self
            animate = visualizer.train_plot()
        self.evals_ind = -1
        self.train_start_time = time.time()
        min_obj, min_obj_val = np.Infinity, np.Infinity
        argmin_params, cvrg_iter = self.params, 0
        for iter in range(max_iter):
            self.cross_validate(X, Y, cv_nfolds, iter=iter,
                callback=None if(visualizer is None) else animate)
            if(iter%(max_iter//10) == 1):
                self.echo('-'*26, 'VALIDATION ITERATION', iter, '-'*27)
                self._print_current_evals()
            obj_val = self.evals[obj_type][1][-1]
            if(obj_val < min_obj_val):
                if(min_obj_val-obj_val < cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_obj = self.evals['obj'][1][-1]
                min_obj_val = obj_val
                self.evals_ind = len(self.evals['obj'][1])-1
                argmin_params = self.params.copy()
            else:
                cvrg_iter += 1
            if(iter > 30 and cvrg_iter > max_cvrg):
                break
            elif(cvrg_iter > max_cvrg*0.5):
                randp = np.random.rand()*cvrg_iter/max_cvrg*0.5
                self.params = (1-randp)*self.params+randp*argmin_params
        self.params = argmin_params.copy()
        self.cross_validate(X, Y, cv_nfolds, False)
        self.evals_ind = -1
        verbose = self.verbose
        self.verbose = True
        self.echo('-'*29, 'OPTIMIZATION RESULT', '-'*30)
        self._print_current_evals()
        self.verbose = verbose

    def predict(self, Xs):
        self.Xs = self.scaler['X'].transform(Xs.astype(self.float_type))
        pred_inputs = self.pack_pred_func_inputs(self.Xs)
        predicted_mats = self.compiled_funcs['pred'](*pred_inputs)
        self.predicted_mats = self.unpack_predicted_mats(predicted_mats)
        mu_fs = self.predicted_mats['mu_Fs']
        std_fs = self.predicted_mats['std_Fs']
        mu_Ys = self.scaler['Y'].recover(mu_fs)
        up_bnd_Ys = self.scaler['Y'].recover(mu_fs+std_fs)
        dn_bnd_Ys = self.scaler['Y'].recover(mu_fs-std_fs)
        std_Ys = 0.5*(up_bnd_Ys-dn_bnd_Ys)
        return mu_Ys, std_Ys

    def get_vars_for_prediction(self):
        return ['D', 'H', 'archit',
                'scaler',
                'params',
                'trained_mats',
                'compiled_funcs',
                'evals_ind',
                'evals']

    def save(self, path):
        import pickle
        save_vars = self.get_vars_for_prediction()
        save_dict = {varn: self.__dict__[varn] for varn in save_vars}
        sys.setrecursionlimit(10000)
        with open(path, 'wb') as save_f:
            pickle.dump(save_dict, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, 'rb') as load_f:
            load_dict = pickle.load(load_f)
        for varn, var in load_dict.items():
            self.__dict__[varn] = var
        return self

    def _print_current_evals(self):
        for metric in sorted(self.evals.keys()):
            eval = self.evals[metric][1][self.evals_ind]
            model_name = self.__str__()
            float_len = 64-len(model_name) if eval > 0 else 63-len(model_name)
            aligned = ('%6s = %.'+str(float_len)+'e')%(metric, eval)
            self.echo(model_name, aligned)

    def _print_evals_comparison(self, evals):
        verbose = self.verbose
        self.verbose = True
        self.echo('-'*30, 'COMPARISON RESULT', '-'*31)
        for metric in sorted(self.evals.keys()):
            eval1 = self.evals[metric][1][self.evals_ind]
            eval2 = evals[metric][1][-1]
            model_name = self.__str__()
            float_len = 27-len(model_name)//2
            float_len1 = float_len-1 if eval1 < 0 else float_len
            float_len2 = float_len-1 if eval2 < 0 else float_len
            aligned = ('%6s = %.'+str(float_len1)+'e <> '+'%.'+str(
                float_len2-len(model_name)%2)+'e')%(metric, eval1, eval2)
            self.echo(model_name, aligned)
        self.verbose = verbose
    
    
    