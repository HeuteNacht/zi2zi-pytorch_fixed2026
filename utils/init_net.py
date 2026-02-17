import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    # 【核心修复】将数字 0 转换为 'cuda:0'
    if len(gpu_ids) > 0 and str(gpu_ids[0]) != '-1':
        device_id = gpu_ids[0]
        device = torch.device(f'cuda:{device_id}')
        net.to(device)
        # 如果有多块显卡
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    else:
        net.to(torch.device('cpu'))
    
    init_weights(net, init_type, init_gain=init_gain)
    return net
