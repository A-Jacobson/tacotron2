import numpy as np


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class SGDRScheduler:
    """
    Implements STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS (SGDR)
    with cosine annealing from https://arxiv.org/pdf/1608.03983.pdf.
    """

    def __init__(self, optimizer, min_lr, max_lr, cycle_length, warmup_steps=5, current_step=0):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr = optimizer.param_groups[0]['lr']
        self.cycle_length = cycle_length
        self.current_step = current_step
        self.warmup_steps = warmup_steps

    def calculate_lr(self):
        """
        calculates new learning rate with cosine annealing
        """
        step = self.current_step % self.cycle_length  # get step in current cycle
        self.lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                  (1 + np.cos((step / self.cycle_length) * np.pi))

    def step(self):
        self.current_step += 1
        self.calculate_lr()
        if self.current_step in range(self.warmup_steps):
            adjust_lr(self.optimizer, self.lr / 10.0)  # warmup with lower lr
        else:
            adjust_lr(self.optimizer, self.lr)


class LRFinderScheduler:
    """
    Implements exponential learning rate finding schedule from
    STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS (SGDR)
    https://arxiv.org/pdf/1608.03983.pdf.

    Increases the learning rate exponentially every step.
    Plot loss vs learning rate and choose rate at which loss was decreases the most quickly.

    """

    def __init__(self, optimizer, min_lr=1e-6, gamma=2.5, current_step=0):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.gamma = gamma
        self.lr = optimizer.param_groups[0]['lr']
        self.current_step = current_step

    def calculate_lr(self):
        self.lr = self.min_lr * (self.current_step ** self.gamma)

    def step(self):
        self.current_step += 1
        self.calculate_lr()
        adjust_lr(self.optimizer, self.lr)
