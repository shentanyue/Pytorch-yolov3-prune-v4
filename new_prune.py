import torch
import argparse
from models import *
import torch.nn.functional as F
import os
import test
from utils import utils


def arg_parse():
    parser = argparse.ArgumentParser(description="YOLO v3 Prune")
    parser.add_argument("--cfg", dest="cfgfile", help="网络模型",
                        default='./cfg/yolov3.cfg', type=str)
    parser.add_argument("--weights", dest="weightsfile", help="权重文件",
                        default='./weights/latest.weights', type=str)
    parser.add_argument('--percent', type=float, default=0.1, help='剪枝的比例')
    return parser.parse_args()




# alpha = 0.1
args = arg_parse()
start = 0
CUDA = torch.cuda.is_available()
print("load network")
model = Darknet(args.cfgfile)
print("done!")
print("load weightsfile")

best_weights_file = './weights/latest.weights'
checkpoint = torch.load(best_weights_file, map_location='cpu')
model.load_state_dict(checkpoint['model'])

# model.load_weights(args.weightsfile)

if CUDA:
    model.cuda()

nnlist = model.module_list
total = 0
donntprune = utils.dontprune(model)
for i in range(len(nnlist)):
    if 'conv_with_bn' in list(nnlist[i].named_children())[0][0]:
        if i not in donntprune:
            total += list(nnlist[i].named_children())[1][1].weight.data.shape[0]
bn = torch.zeros(total)
index = 0
for i in range(len(nnlist)):
    if 'conv_with_bn' in list(nnlist[i].named_children())[0][0]:
        if i not in donntprune:
            size = list(nnlist[i].named_children())[1][1].weight.data.shape[0]
            bn[index:(index + size)] = list(nnlist[i].named_children())[1][1].weight.data.abs().clone()
            index += size
y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index].cuda()
# print(y)
pruned = 0
cfg = []
cfg_mask = []
# print(thre)
print('--' * 30)
print("Pre-processing...")
# 处理bias值
remain_bias_list = dict()
for i in range(len(nnlist)):
    if i not in donntprune:
        for name in nnlist[i].named_children():
            if "_".join(name[0].split("_")[0:-1]) == 'batch_norm':
                weight_copy = name[1].weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()  # 掩模
                if int(torch.sum(mask)) == 0:  # 如果该层所有都被剪掉的时候
                    mask[int(torch.argmax(weight_copy))] = 1.
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                name[1].weight.data.mul_(mask)  # 直接修改γ，
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(i, mask.shape[0], int(torch.sum(mask))))
                bias_mask = torch.ones_like(mask) - mask
                remain_bias = bias_mask * name[1].bias.data
                remain_bias_list[i] = remain_bias
                for next_name in nnlist[i + 1].named_children():
                    if "_".join(next_name[0].split("_")[0:-1]) == 'conv_with_bn':
                        activations = torch.mm(F.relu(remain_bias).view(1, -1),
                                               next_name[1].weight.data.sum(dim=[2, 3]).transpose(1, 0).contiguous())
                        mean = nnlist[i + 1][1].running_mean - activations
                        mean = mean.view(-1)
                        nnlist[i + 1][1].running_mean = mean
                    elif "_".join(next_name[0].split("_")[0:-1]) == 'conv_without_bn':
                        activations = torch.mm(F.relu(remain_bias).view(1, -1),
                                               next_name[1].weight.data.sum(dim=[2, 3]).transpose(1, 0).contiguous())
                        bias = next_name[1].bias.data + activations
                        bias = bias.view(-1)
                        next_name[1].bias.data = bias
                    # elif next_name[0].split("_")[0] == 'maxpool':
                    #     activations = torch.mm(F.relu(remain_bias).view(1, -1),
                    #                            nnlist[i + 2][0].weight.sum(dim=[2, 3]).transpose(1, 0).contiguous())
                    #     mean = nnlist[i + 2][1].running_mean - activations
                    #     mean = mean.view(-1)
                    #     nnlist[i + 2][1].running_mean = mean
                    # elif next_name[0].split("_")[0] == 'reorg':
                    #     stride = next_name[1].stride
                    #     remain_bias_list[i + 1] = torch.squeeze(
                    #         remain_bias.expand(int(stride * stride), int(remain_bias.size(0))).transpose(1,
                    #                                                                                      0).contiguous().view(
                    #             1, -1))
            elif name[0].split("_")[0] == 'route':
                try:
                    end = name[1].layer[1]
                except:
                    end = 0
                prev_1 = name[1].layers[0] + i
                have_prev_2 = False
                # print(name[1])
                # print(name[1].layers)
                if end != 0:
                    prev_2 = name[1].layers[1] + i
                    have_prev_2 = True
                if isinstance(nnlist[prev_1][0], nn.Conv2d):
                    if not have_prev_2:
                        remain_bias = remain_bias_list[prev_1]
                    else:
                        remain_bias = torch.cat((remain_bias_list[prev_1], remain_bias_list[prev_2]), 0)
                    activations = torch.mm(F.relu(remain_bias).view(1, -1),
                                           nnlist[i + 1][0].weight.sum(dim=[2, 3]).transpose(1, 0).contiguous())
                    mean = nnlist[i + 1][1].running_mean - activations
                    mean = mean.view(-1)
                    nnlist[i + 1][1].running_mean = mean
    else:
        for name in nnlist[i].named_children():
            if "_".join(name[0].split("_")[0:-1]) == 'batch_norm':
                dontp = name[1].weight.data.numel()
                mask = torch.ones(name[1].weight.data.shape)
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(i, dontp, int(dontp)))
                cfg.append(int(dontp))
                cfg_mask.append(mask.clone())

print(cfg)
pruned_ratio = pruned / total
print('Pre-processing Successful!')
print('--' * 30)
# print(cfg)
# 写出被减枝的cfg文件
prunecfg = utils.write_cfg(args.cfgfile, cfg, args.percent)
newmodel = Darknet(prunecfg)
newmodel.header_info = model.header_info
if CUDA:
    newmodel.cuda()
old_modules = list(model.modules())
new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
print("pruning...")
v = 0
for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):  # 向新模型中写入
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()  # 去掉的bias导致精度大量下降
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Sequential):
        for name in m0.named_children():
            if name[0].split("_")[0] == 'route':
                ind = v + old_modules[layer_id + 1].layers[0]
                # print('ind:',ind)
                cfg_mask1 = cfg_mask[utils.route_problem(model, ind)]
                try:
                    end = old_modules[layer_id + 1].layers[1]
                except:
                    end = 0
                if end != 0:
                    ind = old_modules[layer_id + 1].layers[1]
                    # print('ind2:',ind)
                    # print('v:',v)
                    # print('old_modules[layer_id + 1].layers[1]:',old_modules[layer_id + 1].layers[1])
                    cfg_mask1 = cfg_mask1.unsqueeze(0)
                    cfg_mask2 = cfg_mask[utils.route_problem(model, ind)].unsqueeze(0).cuda()
                    cfg_mask3 = torch.cat((cfg_mask1, cfg_mask2), 1)
                    cfg_mask1 = cfg_mask3.squeeze(0)
                start_mask = cfg_mask1.clone()
            elif "_".join(name[0].split("_")[0:-1]) == 'conv_with_bn':
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('Conv In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = old_modules[layer_id + 1].weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                new_modules[layer_id + 1].weight.data = w1.clone()
            elif "_".join(name[0].split("_")[0:-1]) == 'conv_without_bn':
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                w1 = old_modules[layer_id + 1].weight.data[:, idx0.tolist(), :, :].clone()
                new_modules[layer_id + 1].weight.data = w1.clone()
                new_modules[layer_id + 1].bias.data = old_modules[layer_id + 1].bias.data.clone()
                print('Detect: In shape: {:d}, Out shape {:d}.'.format(new_modules[layer_id + 1].weight.data.size(1),
                                                                       new_modules[layer_id + 1].weight.data.size(0)))
        v = v + 1
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()
print('--' * 30)
print('prune done!')
print('pruned ratio %.3f' % pruned_ratio)
prunedweights = os.path.join("prune_{}_".format(args.percent) + args.weightsfile.split("/")[-1])
print('save weights file in %s' % prunedweights)
if not os.path.exists('prune_weights'):
    os.mkdir('prune_weights')
newmodel.save_weights('prune_weights/'+prunedweights)

net_config_path = './prune_cfg/prune_{}_yolov3.cfg'.format(args.percent)
data_config_path = 'cfg/coco.data'
latest_weights_file = 'prune_weights/' + prunedweights
mAP, R, P = test.test(
    net_config_path,
    data_config_path,
    latest_weights_file,
    batch_size=16,
    img_size=416,
    conf_thres=0.1
)
print('done!')
