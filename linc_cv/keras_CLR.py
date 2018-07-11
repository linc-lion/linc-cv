import keras.backend as K
import numpy as np
from keras.callbacks import Callback


class CLR(Callback):
    """This callback implements a cyclical learning rate and momentum policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration 
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CLR(min_lr=1e-3, max_lr=1e-2,
                      min_mtm=0.85, max_mtm=0.95,
                      annealing=0.1,num_steps=np.ceil((X_train.shape[0]*epochs/batch_size)))
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    # Arguments
        min_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - min_lr).
        num_steps: number of training iterations in the cycle. To define as `np.ceil((X_train.shape[0]*epochs/batch_size))`
        max_mtm : initial value of the momentum    
        min_mtm : lower boundary in the cycle.
        annealing : percentage of the iterations where the lr
                    will decrease lower than its min_lr
                    
        # References
        Original paper: https://arxiv.org/pdf/1803.09820.pdf
        Inspired by : https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy

    """

    def __init__(self, min_lr=1e-5, max_lr=1e-2, min_mtm=0.85, max_mtm=0.95, num_steps=1000, annealing=0.1):
        super(CLR, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mtm = min_mtm
        self.max_mtm = max_mtm
        self.annealing = annealing
        self.num_steps = num_steps
        self.an_step = int(np.ceil(num_steps * (1 - annealing)))
        self.iterations = 0
        self.history = {}

    def clr(self):
        if self.iterations >= self.an_step:
            lr = -self.min_lr * (self.num_steps - self.iterations) / (self.an_step - self.num_steps)
        else:
            lr = self.min_lr - (self.max_lr - self.min_lr) * (2 * np.abs(
                (self.num_steps * self.iterations - (self.an_step * self.num_steps) // 2) / (
                        self.an_step * self.num_steps)) - 1)
        return lr

    def cmtm(self):
        if self.iterations >= self.an_step:
            mtm = self.max_mtm
        else:
            mtm = self.min_mtm + 2 * (self.max_mtm - self.min_mtm) * np.abs(
                (self.num_steps * self.iterations - (self.an_step * self.num_steps // 2)) / (
                        self.an_step * self.num_steps))
        return mtm

    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        K.set_value(self.model.optimizer.momentum, self.max_mtm)

    def on_batch_end(self, batch, logs=None):

        logs = logs or {}
        self.iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        clr = self.clr()
        cmtm = self.cmtm()
        K.set_value(self.model.optimizer.lr, clr)
        K.set_value(self.model.optimizer.momentum, cmtm)
        print(f'\nlearning rate -> {clr}, momentum -> {cmtm}')
