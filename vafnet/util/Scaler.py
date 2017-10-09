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


import numpy as np

__all__ = [
    "Scaler"
]

class Scaler(object):

    algos = [
        "min-max",
        "normal",
        "inv-normal",
    ]
    
    data = {}
    
    def __init__(self, algo):
        assert algo.lower() in self.algos, "Invalid Scaling Algorithm!"
        self.algo = algo.lower()
        if(self.algo == "min-max"):
            self.data = {"skip_cols": [], "min": 0, "max":0}
        elif(self.algo == "normal"):
            self.data = {"skip_cols": [], "std": 0, "mu":0}
        elif(self.algo == "inv-normal"):
            self.data = {"skip_cols": [], "std": 0, "mu":0}
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def fit(self, X):
        if(self.algo == "min-max"):
            self.data['min'] = np.min(X, axis=1)[:, None]
            self.data['max'] = np.max(X, axis=1)[:, None]
            self.data['skip_cols'] = np.where(self.data['min']==self.data['max'])[0]
            use_cols = np.delete(np.arange(X.shape[0]), self.data['skip_cols'])
            self.data['min'] = self.data['min'][use_cols]
            self.data['max'] = self.data['max'][use_cols]
        elif(self.algo == "normal"):
            self.data['mu'] = np.mean(X, axis=1)[:, None]
            self.data['std'] = np.std(X, axis=1)[:, None]
            self.data['skip_cols'] = np.where(self.data['std']==0)[0]
            use_cols = np.delete(np.arange(X.shape[0]), self.data['skip_cols'])
            self.data['mu'] = self.data['mu'][use_cols]
            self.data['std'] = self.data['std'][use_cols]
        elif(self.algo == "inv-normal"):
            self.data['mu'] = np.mean(X, axis=1)[:, None]
            self.data['std'] = np.std(X, axis=1)[:, None]
            self.data['skip_cols'] = np.where(self.data['std']==0)[0]
            use_cols = np.delete(np.arange(X.shape[0]), self.data['skip_cols'])
            self.data['mu'] = self.data['mu'][use_cols]
            self.data['std'] = self.data['std'][use_cols]
    
    def transform(self, X):
        use_cols = np.delete(np.arange(X.shape[0]), self.data['skip_cols'])
        if(self.algo == "min-max"):
            return (X[use_cols]-self.data["min"])/(self.data["max"]-self.data["min"])
        elif(self.algo == "normal"):
            return (X[use_cols]-self.data["mu"])/self.data["std"]
        elif(self.algo == "inv-normal"):
            return norm.cdf((X[use_cols]-self.data["mu"])/self.data["std"])
    
    def recover(self, X):
        use_cols = np.delete(np.arange(X.shape[0]), self.data['skip_cols'])
        if(self.algo == "min-max"):
            return X[use_cols]*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.algo == "normal"):
            return X[use_cols]*self.data["std"]+self.data["mu"]
        elif(self.algo == "inv-normal"):
            return (norm.ppf(X[use_cols])-self.data["mu"])/self.data["std"]
    
