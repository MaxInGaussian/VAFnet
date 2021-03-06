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
    "Metric"
]

class Metric(object):
    
    metrics = [
        "mse",
        "nmse",
        "mae",
        "nlpd",
        "nlpd_mse",
        "nlpd_nmse",
        "nlml"
    ]
    
    def __init__(self, metric, gp=None):
        assert metric in self.metrics, "Invalid metric!"
        self.metric = metric  
        self.gp = gp

    def eval(self, target, mu_pred, std_pred):
        return getattr(self, self.metric)(target, mu_pred, std_pred)

    def mse(self, target, mu_pred, std_pred):        
        mse_real = np.mean(np.real(target-mu_pred)**2)
        mse_imag = np.mean(np.imag(target-mu_pred)**2)
        return mse_real/2+mse_imag/2

    def nmse(self, target, mu_pred, std_pred):
        mse_real = np.mean(np.real(target-mu_pred)**2)
        nmse_real = mse_real/np.var(np.real(target))
        if(np.var(np.imag(target)) > 0):
            mse_imag = np.mean(np.imag(target-mu_pred)**2)
            nmse_imag = mse_imag/np.var(np.imag(target))
            return nmse_real/2+nmse_imag/2
        return nmse_real

    def mae(self, target, mu_pred, std_pred):
        mae = np.mean(np.abs(target.real-mu_pred.real))+\
            np.mean(np.abs(target.imag-mu_pred.imag))
        return mae/2

    def nlpd(self, target, mu_pred, std_pred):        
        nlpd = np.mean(((target-mu_pred)/std_pred)**2+2*np.log(std_pred))
        nlpd = 0.5*(np.log(2*np.pi)+nlpd)
        return np.absolute(nlpd)

    def nlpd_mse(self, target, mu_pred, std_pred):        
        return self.mse(target, mu_pred, std_pred)-\
            np.exp(-self.nlpd(target, mu_pred, std_pred))

    def nlpd_nmse(self, target, mu_pred, std_pred):        
        return self.nmse(target, mu_pred, std_pred)-\
            np.exp(-self.nlpd(target, mu_pred, std_pred))
    
    def nlml(self, target, mu_pred, std_pred):
        if(self.gp.mean_only):
            return self.mse(target, mu_pred, std_pred)
        noise = self.gp.noise_real+self.gp.noise_imag*1j
        goodness_of_fit = (target.conj().T.dot(target-mu_pred))/noise
        covariance_penalty = np.sum(np.log(np.diagonal(self.gp.T)))
        noise_penalty = (self.gp.N-self.gp.M)*np.log(noise)
        nlml = goodness_of_fit+covariance_penalty+noise_penalty
        return np.absolute(nlml[0, 0])
