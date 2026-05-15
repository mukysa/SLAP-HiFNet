import torch
import matplotlib.pyplot as plt
import math
import os

class WarmUpScheduler:
    def __init__(self, optimizer, base_lr, max_iters, warmup_iters, power=2.0):
        """
        学习率调度器，支持预热和余弦退火
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power

    def step(self, cur_iters):
        """
        更新学习率
        """
        if cur_iters < self.warmup_iters:
            lr = self.base_lr * cur_iters / (self.warmup_iters + 1e-8)
        else:
            lr = self.base_lr * ((1 - float(cur_iters - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power)
        self.optimizer.param_groups[0]['lr'] = lr
        return lr
    
    
class WarmUpCosineAnnealingScheduler:
    def __init__(self, optimizer, base_lr, min_lr, max_iters, warmup_iters=0):
        """
        余弦退火学习率调度器，支持预热
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters

    def step(self, cur_iters):
        """
        更新学习率
        """
        if cur_iters < self.warmup_iters:
            # 预热阶段，线性增加学习率
            lr = self.base_lr * cur_iters / (self.warmup_iters + 1e-8)
        else:
            # 余弦退火阶段
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(math.pi * (cur_iters - self.warmup_iters) / (self.max_iters - self.warmup_iters)))).item()
        
        self.optimizer.param_groups[0]['lr'] = lr
        return lr


def test_learning_rate_adjustment(train_options):
    """
    模拟训练，可视化学习率变化
    """
    # 设置训练参数
    max_epochs = train_options['epochs']
    iters_per_epoch = train_options['epoch_len']
    warmup_epochs = train_options['warmup_epochs']

    # 创建一个假的优化器用于学习率调整
    dummy_optimizer = torch.optim.SGD([torch.randn(10)], lr=train_options['base_lr'], momentum=train_options['momentum'], weight_decay=train_options['weight_decay'])
    # dummy_optimizer = torch.optim.Adam([torch.randn(10)], lr=train_options['base_lr'], momentum=train_options['momentum'], weight_decay=train_options['weight_decay'])

    # 初始化学习率调度器
    max_iters = max_epochs * iters_per_epoch
    warmup_iters = warmup_epochs * iters_per_epoch if warmup_epochs > 0 else 0
    # scheduler = WarmUpScheduler(dummy_optimizer, train_options['base_lr'], max_iters, warmup_iters)
    scheduler = WarmUpCosineAnnealingScheduler(dummy_optimizer, train_options['base_lr'], train_options['min_lr'], max_iters, warmup_iters)

    # 记录学习率变化
    learning_rates = []

    # 模拟训练过程
    for epoch in range(max_epochs):
        for iter in range(iters_per_epoch):
            cur_iters = epoch * iters_per_epoch + iter
            lr = scheduler.step(cur_iters)
            learning_rates.append(lr)

    # 可视化学习率变化并保存
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Adjustment')
    plt.legend()
    plt.grid()
    plt.savefig('learning_rate_curve.png')


if __name__ == '__main__':
    # 设置训练选项
    train_options = {
        # 优化器设置
        'base_lr': 1e-5,  # 基础学习率
        'momentum': 0.9,
        'weight_decay': 1e-3,
        # ws
        'warmup_epochs': 5,  # 热身轮数
        'min_lr': 0,  # 最小学习率
        # epochs
        'epochs': 60,
        'epoch_len': 500
    }

    # 测试学习率调整
    test_learning_rate_adjustment(train_options)