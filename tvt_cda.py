import argparse
from ast import arg
from inspect import isgetsetdescriptor
import os, sys
from pyexpat import features

from sqlalchemy import false

sys.path.append('./')

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.modeling_orig import CONFIGS, VisionTransformer_DomainClassifier_cda, VAE
from models.lossZoo import im
from torch.optim.lr_scheduler import LambdaLR
import math, random
from data_impression import loss_function
from utils import print_args
from my_loader import officeHome_load, office31_load, visda_load
import network
import joblib
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc


def vae_test(model, generator, test_loader):
    model.eval()
    generator.eval()
    loss = [0, 0, 0]
    epoch_iterator = iter(test_loader)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)
        x, y, _ = batch
        with torch.no_grad():
            feature = model.freeze_netF(x)
            feature = feature.detach().view(-1, 3, 16, 16).cuda()
            results = generator.forward(feature)
            generator_loss = generator.loss_function(*results,
                                                     M_N=1,  # al_img.shape[0]/ self.num_train_imgs,
                                                     )

        loss[0] += generator_loss['loss']
        loss[1] += generator_loss['Reconstruction_Loss']
        loss[2] += generator_loss['KLD']
    loss = [i / len(test_loader.dataset) for i in loss]

    return loss


def cal_acc(args, model, test_loader, s=100, t=0):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = iter(test_loader)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.cuda() for t in batch)
        x, y, _ = batch
        with torch.no_grad():
            output, _ = model.netB(model.netF(x), s=s, t=t)
            logits = model.netC(output)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )

    all_preds, all_label = all_preds[0], all_label[0]
    print(len(all_preds))
    if args.dset == 'visda':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = (all_preds == all_label).mean()

    if args.dset == 'visda':
        return accuracy, classWise_acc
    else:
        return accuracy, None


def cal_acc_domain_agnostic(args, loader, model, t):
    model.eval()
    # d_cls.eval()
    start_test = True
    all_preds, all_label = [], []
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            out = model.freeze_netF(inputs)
            out = model.netD(out, args=None, test=True)
            x = model.netF(inputs)
            outputs, _ = model.netB(x, s=args.SMAX, t=t, d_cls_out=out, type='hard_mask')
            batch_outputs = model.netC(outputs)
            pred_domain_ = torch.argmax(out, dim=-1).detach().cpu().numpy()
            if i == 0:
                pred_domain = pred_domain_
            else:
                pred_domain = np.append(pred_domain, pred_domain_)
            # print(batch_outputs.shape)
            preds = torch.argmax(batch_outputs, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], labels.detach().cpu().numpy(), axis=0
                )
    all_preds, all_label = all_preds[0], all_label[0]
    if args.dset == 'visda':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = (all_preds == all_label).mean()
    domain_acc = (pred_domain == t).mean()
    from collections import Counter
    print(Counter(pred_domain))
    if args.dset == 'visda':
        return accuracy, classWise_acc
    else:
        return accuracy, domain_acc


def init_bank(num_sample, dset_loader, model, args, t):
    fea_bank = torch.randn(num_sample, 768)
    score_bank = torch.randn(num_sample, args.class_num)
    label_bank_raw = torch.zeros(num_sample, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        iter_test = iter(dset_loader)
        for i in range(len(dset_loader)):
            data = iter_test.next()
            inputs = data[0]
            indx = data[-1]
            labels = data[1]
            inputs = inputs.cuda()
            output, _ = model.netB(model.netF(inputs), t=t)
            output_norm = torch.nn.functional.normalize(output)
            outputs = model.netC(output)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = output_norm.detach().cpu()
            score_bank[indx] = outputs.detach().cpu()
            label_bank_raw[indx] = labels
    label_bank = torch.eye(args.class_num)[label_bank_raw]
    return fea_bank, score_bank, label_bank, label_bank_raw


def train_source(args, dset_loaders):
    # 源域训练集：测试集8：2，目标域训练集测试集相同，全部数据
    ## set base network
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer_DomainClassifier_cda(config, args.img_size, zero_head=True, num_classes=args.class_num,
                                                   bottleneck=args.bottleneck, freeze_layers=args.freeze_layer)
    model.load_from(np.load("checkpoint/ViT-B_16.npz"))
    model = model.cuda()

    args.gradient_accumulation_steps = 1
    args.train_batch_size = args.batch_size // args.gradient_accumulation_steps
    train_loader = dset_loaders['0tr']

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=3e-2,
                                momentum=0.9,
                                weight_decay=0)
    t_total = 5000
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=t_total)
    model.zero_grad()
    global_step, best_acc = 0, 0
    loss_function = torch.nn.CrossEntropyLoss()
    smax = 100

    while True:
        model.train()
        iter_source = iter(train_loader)
        for step, batch in enumerate(iter_source):
            batch = tuple(t.cuda() for t in batch)
            x, y, _ = batch
            progress_ratio = step / (len(train_loader) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax

            outputs, masks = model.netB(model.netF(x), s=s, t=0, all_out=True)
            output = [model.netC(outputs[i]) for i in range(args.task_num)]
            reg = 0
            count = 0
            for i in range(args.task_num):
                for m in masks[i]:
                    reg += m.sum()  # numerator
                    count += np.prod(m.size()).item()  # denominator

            reg /= count
            loss = 0
            for i in range(args.task_num):
                loss += loss_function(output[i], y)
            loss += 0.1 * reg

            optimizer.zero_grad()
            loss.backward()

            # Compensate embedding gradients
            for n, p in model.em.named_parameters():
                num = torch.cosh(torch.clamp(s * p.data, -10, 10)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 10000)
            global_step += 1
            if global_step % 500 == 0:
                acc = []
                for i in range(args.task_num):
                    accuracy, _ = cal_acc(args, model, dset_loaders[str(i) + 'ts'], t=i)
                    acc.append(accuracy)
                if args.task_num == 4:
                    log_str = 'Task: {}, Epoch: {}/{}, loss:{:.4f}; Accuracy on 4 domains = {:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%.'.format(
                        t, global_step, t_total, loss, acc[0] * 100, acc[1] * 100, acc[2] * 100, acc[3] * 100)
                if args.task_num == 3:
                    log_str = 'Task: {}, Epoch: {}/{}, loss:{:.4f}; Accuracy on 3 domains = {:.2f}%|{:.2f}%|{:.2f}%.'.format(
                        t, global_step, t_total, loss, acc[0] * 100, acc[1] * 100, acc[2] * 100)
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str)
                model.train()
            if global_step % t_total == 0:
                break

        if global_step % t_total == 0:
            break
    torch.save(model.state_dict(), osp.join(args.model_dir, "source_{}.pt".format(args.max_epoch)))
    return model


def train_source_g(args, dset_loaders):
    # 源域训练集：测试集8：2，目标域训练集测试集相同，全部数据
    ## set base network
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer_DomainClassifier_cda(config, args.img_size, zero_head=True, num_classes=args.class_num,
                                                   bottleneck=args.bottleneck, freeze_layers=args.freeze_layer)
    model.load_from(np.load("checkpoint/ViT-B_16.npz"))
    model = model.cuda()

    generator = VAE().cuda()

    args.gradient_accumulation_steps = 1
    args.train_batch_size = args.batch_size // args.gradient_accumulation_steps
    train_loader = dset_loaders['0tr']

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=3e-2,
                                momentum=0.9,
                                weight_decay=0)
    t_total = 2000
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=t_total)

    optimizer_g = optim.Adam(generator.parameters(),
                             lr=0.005,
                             weight_decay=0.0)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.95)

    model.zero_grad()
    generator.zero_grad()
    global_step, best_acc = 0, 0
    loss_function = torch.nn.CrossEntropyLoss()
    smax = 100

    while True:
        model.train()
        iter_source = iter(train_loader)
        for step, batch in enumerate(iter_source):
            batch = tuple(t.cuda() for t in batch)
            x, y, _ = batch
            progress_ratio = step / (len(train_loader) - 1)
            s = (smax - 1 / smax) * progress_ratio + 1 / smax

            outputs, masks = model.netB(model.netF(x), s=s, t=0, all_out=True)
            output = [model.netC(outputs[i]) for i in range(args.task_num)]
            reg = 0
            count = 0
            for i in range(args.task_num):
                for m in masks[i]:
                    reg += m.sum()  # numerator
                    count += np.prod(m.size()).item()  # denominator

            reg /= count
            loss = 0
            for i in range(args.task_num):
                loss += loss_function(output[i], y)
            loss += 0.1 * reg

            optimizer.zero_grad()
            loss.backward()

            # Compensate embedding gradients
            for n, p in model.em.named_parameters():
                num = torch.cosh(torch.clamp(s * p.data, -10, 10)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= smax / s * num / den
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm(model.parameters(), 10000)
            optimizer.step()
            scheduler.step()

            feature = model.freeze_netF(x)
            feature = feature.detach().view(-1, 3, 16, 16).cuda()
            results = generator.forward(feature)
            generator_loss = generator.loss_function(*results,
                                                     M_N=0.00025,  # al_img.shape[0]/ self.num_train_imgs,
                                                     )
            loss_g = generator_loss['loss']

            optimizer_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10000)
            optimizer_g.step()
            scheduler_g.step()

            global_step += 1
            if global_step % 500 == 0:
                model.eval()
                generator.eval()
                acc = []
                for i in range(args.task_num):
                    accuracy, _ = cal_acc(args, model, dset_loaders[str(i) + 'ts'], t=i)
                    acc.append(accuracy)
                if args.task_num == 4:
                    log_str = 'Task: {}, Epoch: {}/{}, loss:{:.4f}; Accuracy on 4 domains = {:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%.'.format(
                        t, global_step, t_total, loss, acc[0] * 100, acc[1] * 100, acc[2] * 100, acc[3] * 100)
                if args.task_num == 3:
                    log_str = 'Task: {}, Epoch: {}/{}, loss:{:.4f}; Accuracy on 3 domains = {:.2f}%|{:.2f}%|{:.2f}%.'.format(
                        t, global_step, t_total, loss, acc[0] * 100, acc[1] * 100, acc[2] * 100)
                # args.out_file.write(log_str + '\n')
                # args.out_file.flush()
                # print(log_str)
                model.train()
            if global_step % t_total == 0:
                break

        if global_step % t_total == 0:
            break
    torch.save(model.state_dict(), osp.join(args.model_dir, "source_{}.pt".format(0)))
    torch.save(generator.state_dict(), osp.join(args.model_dir, "generator_{}.pt".format(0)))
    return model, generator


def cvae_test(args, model, mnist_test, netF):
    test_avg_loss = 0.0
    netF.eval()
    with torch.no_grad():  # 这一部分不计算梯度，也就是不放入计算图中去
        '''测试测试集中的数据'''
        # 计算所有batch的损失函数的和
        for test_batch_index, (test_x, label, _) in enumerate(iter(mnist_test)):
            test_x = test_x.cuda()
            outputs = netF.freeze_netF(test_x)
            # bias = torch.ones(outputs.shape).long().cuda()
            # data = outputs+bias
            data = outputs.cuda()
            # 前向传播
            test_x_hat, test_mu, test_log_var = model(data, label)
            # 损害函数值
            flat_data = data.view(-1, data.shape[1])

            y_condition = model.to_categrical(label).cuda()
            con = torch.cat((flat_data, y_condition), 1)
            test_loss, test_BCE, test_KLD = loss_function(test_x_hat, con, test_mu, test_log_var)
            test_avg_loss += test_loss

        # 对和求平均，得到每一张图片的平均损失
        test_avg_loss /= len(mnist_test.dataset)
        # print('测试集输出特征\n')
        # print(outputs)
        '''测试随机生成的隐变量'''
        # 随机从隐变量的分布中取隐变量
        z = torch.randn(64, args.z_dim).cuda()  # 每一行是一个隐变量，总共有batch_size行
        c = np.zeros(shape=(z.shape[0],))
        rand = np.random.randint(0, args.class_num)
        print(f"Random number: {rand}")
        c[:] = rand
        c = torch.FloatTensor(c)
        # 对隐变量重构
        random_res = model.decode(z, c).cpu()
        # 模型的输出矩阵：每一行的末尾都加了one-hot向量，要去掉这个one-hot向量
        generated_image = random_res[:, 0:random_res.shape[1] - args.class_num]
        # bias = torch.ones(generated_image.shape).long()
        # gen = generated_image /10
        # gen = inversesigmoid(generated_image)
        # print(gen.shape)
        print(generated_image)

        return test_avg_loss


def generate_feature(args, dset_loaders, t, netF):
    # Step 1: 载入数据
    # mnist_test, mnist_train, classes = dataloader(args.batch_size, args.num_worker)
    train_dataset = dset_loaders[str(t) + 'gr']
    test_dataset = dset_loaders[str(t) + 'gr']

    # 查看每一个batch图片的规模
    x, label, _ = iter(train_dataset).__next__()  # 取出第一批(batch)训练所用的数据集
    print(' img : ', x.shape)  # img :  torch.Size([batch_size, 1, 28, 28])， 每次迭代获取batch_size张图片，每张图大小为(1,28,28)

    # Step 2: 准备工作 : 搭建计算流程
    model = network.CVAE(input_dim=args.input_dim, y_dim=args.class_num, z_dim=args.z_dim).cuda()  # 生成AE模型，并转移到GPU上去
    print('The structure of our model is shown below: \n')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 生成优化器，需要优化的是model的参数，学习率为0.001

    # Step 3: optionally resume(恢复) from a checkpoint
    start_epoch = 0
    best_test_loss = np.finfo('f').max
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         # 载入已经训练过的模型参数与结果
    #         print('=> loading checkpoint %s' % args.resume)
    #         checkpoint = torch.load(args.resume)
    #         start_epoch = checkpoint['epoch'] + 1
    #         best_test_loss = checkpoint['best_test_loss']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print('=> loaded checkpoint %s' % args.resume)
    #     else:
    #         print('=> no checkpoint found at %s' % args.resume)

    # Step 4: 开始迭代
    netF.eval()
    loss_epoch = []
    for epoch in range(start_epoch, args.epochs):

        # 训练模型
        # 每一代都要遍历所有的批次
        loss_batch = []
        for batch_index, (x, label, _) in enumerate(iter(train_dataset)):
            # x : [b, 1, 28, 28], remember to deploy the input on GPU
            if x.size(0) == 1:
                continue
            x = x.cuda()
            outputs = netF.freeze_netF(x)
            data = outputs
            # print(data)
            # print(data.shape)
            data = data.cuda()
            # 前向传播
            x_hat, mu, log_var = model(data, label)  # 模型的输出，在这里会自动调用model中的forward函数
            # 训练样本展平，在每个样本后面连接标签的one-hot向量
            flat_data = data.view(-1, data.shape[1])
            # print(data.shape, flat_data.shape)
            y_condition = model.to_categrical(label).cuda()
            con = torch.cat((flat_data, y_condition), 1)
            loss, BCE, KLD = loss_function(x_hat, con, mu, log_var)  # 计算损失值，即目标函数
            loss_batch.append(loss.item())  # loss是Tensor类型

            # 后向传播
            optimizer.zero_grad()  # 梯度清零，否则上一步的梯度仍会存在
            loss.backward()  # 后向传播计算梯度，这些梯度会保存在model.parameters里面
            optimizer.step()  # 更新梯度，这一步与上一步主要是根据model.parameters联系起来了

            # print statistics every 100 batch
            if (batch_index + 1) % 10 == 0:
                logstr = ('Epoch [{}/{}], Batch [{}/{}] : Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                          .format(epoch + 1, args.epochs, batch_index + 1, len(train_dataset.dataset) // args.batch_size,
                                  loss.item() / args.batch_size, BCE.item() / args.batch_size,
                                  KLD.item() / args.batch_size))
                args.out_file.write(logstr + '\n')
                args.out_file.flush()
                print(logstr)

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(train_dataset.dataset))  # len(mnist_train.dataset)为样本个数

        # 测试模型
        if (epoch + 1) % args.test_every == 0:
            test_loss = cvae_test(args, model, test_dataset, netF)
            logstr = ('Epoch [{}/{}],Test-loss= {:.4f}'.format(epoch + 1, args.epochs, test_loss))
            args.out_file.write(logstr + '\n')
            args.out_file.flush()
            print(logstr)

    store_CVAE = model.state_dict()
    torch.save(store_CVAE, osp.join(args.model_dir, "CVAE_{}.pt".format(t)))
    return loss_epoch


def train_target_near(args, t, model, dset_loaders, netG, ACC_list=None):
    generator = VAE().cuda()
    freeze_layer = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.']
    # print(freeze_layer)
    # for name, para in model.named_parameters():
    #     # 除head, pre_logits外，其他权重全部冻结
    #     if name[:9] in freeze_layer:
    #         para.requires_grad_(False)
    #         # print('freeze {}'.format(name))
    print('Training Task : {}'.format(t))
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,  # args.lr/10
                                momentum=0.9,
                                weight_decay=0)
    optimizer_g = optim.Adam(generator.parameters(),
                             lr=0.005,
                             weight_decay=0.0)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.95)
    generator.zero_grad()

    t_total = 500 * args.grad_iter
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=500)
    model.zero_grad()
    global_step, best_acc = 0, 0

    num_sample = len(dset_loaders[str(t) + 'tr'].dataset)
    feature_bank, score_bank, label_bank, label_bank_raw = init_bank(num_sample, dset_loaders[str(t) + 'tr'], model, args, t)
    few_short_index = []
    if args.shot > 0:
        few_short_index = dset_loaders[str(t) + 'few_short_index']
        few_short_index = torch.from_numpy(few_short_index)
        score_bank[few_short_index] = label_bank[few_short_index]
        unlabel_index = torch.from_numpy(dset_loaders[str(t) + 'unlabel_index'])

    while global_step <= t_total:
        if global_step % 500 == 0 or global_step == t_total:
            model.eval()
            acc = []
            acc1 = []
            acc2 = []
            for i in range(args.task_num):
                accuracy, _ = cal_acc(args, model, dset_loaders[str(i) + 'ts'], t=i)
                acc_agno, doamin_acc = cal_acc_domain_agnostic(args, dset_loaders[str(i) + 'ts'], model, t=i)
                acc.append(accuracy)
                acc1.append(acc_agno)
                acc2.append(doamin_acc)
            if args.task_num == 4:
                log_str = 'Task: {}, Epoch: {}/{}; Accuracy on 4 domains = {:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%({:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%)|({:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%).'.format(
                    t, global_step, t_total, acc[0] * 100, acc[1] * 100, acc[2] * 100, acc[3] * 100, acc1[0] * 100, acc1[1] * 100, acc1[2] * 100,
                                             acc1[3] * 100,acc2[0] * 100, acc2[1] * 100, acc2[2] * 100, acc2[3] * 100)
            if args.task_num == 3:
                log_str = 'Task: {}, Epoch: {}/{}; Accuracy on 3 domains = {:.2f}%|{:.2f}%|{:.2f}%｜({:.2f}%|{:.2f}%|{:.2f}%)|({:.2f}%|{:.2f}%|{:.2f}%).'.format(
                    t, global_step, t_total, acc[0] * 100, acc[1] * 100, acc[2] * 100, acc1[0] * 100, acc1[1] * 100, acc1[2] * 100,acc2[0] * 100, acc2[1] * 100, acc2[2] * 100)
            ACC_list.append(acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            if global_step == t_total:
                break
        model.train()

        flag = 0
        if args.MixMatch:
            try:
                input_X1, input_X2, _, index = batch_iter1.next()
            except:
                batch_iter1 = iter(dset_loaders[str(t) + 'tr_twice_unlabeled'])
                input_X1, input_X2, _, index = batch_iter1.next()
            try:
                input_X, y_labeled, index_labeled = batch_iter2.next()
            except:
                batch_iter2 = iter(dset_loaders[str(t) + 'tr_labeled'])
                input_X, y_labeled, index_labeled = batch_iter2.next()
            index_labeled = few_short_index[index_labeled]
            index = unlabel_index[index]
            if len(input_X1) == args.batch_size:
                input_X1 = torch.cat([input_X, input_X1], dim=0)
                index = torch.cat([index_labeled, index], dim=0)
                flag = 1
        else:
            if global_step % (len(dset_loaders[str(t) + 'tr_twice']) - 1) == 0:
                batch_iter = iter(dset_loaders[str(t) + 'tr_twice'])
            input_X1, input_X2, _, index = batch_iter.next()
        if input_X1.size(0) == 1:
            continue

        global_step += 1

        # lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        input_X1_cuda = input_X1.cuda()
        x = model.freeze_netF(input_X1_cuda)
        task_pre = random.randrange(0, t)

        out, labels, dls_out = model.netD(x, args, task_pre, t, netG[task_pre])
        # print(task_pre, t,labels.shape,out.shape)
        pre_domain = torch.argmax(out, dim=-1).detach().cpu().numpy()
        prob = (pre_domain == torch.argmax(labels, dim=-1).detach().cpu().numpy()).mean()

        args.type = 'soft_mask' if prob < 1.0 else 'hard_mask'
        output_feature, masks = model.netB(model.netF(input_X1_cuda), s=args.SMAX, t=t, d_cls_out=dls_out, type=args.type)
        output_proba1 = nn.Softmax(dim=1)(model.netC(output_feature))
        masks_mean = torch.sum(masks, dim=0) / masks.shape[0]
        masks_old = masks_mean
        with torch.no_grad():
            feature_bank[index].fill_(-0.1)
            output_feature_norm = torch.nn.functional.normalize(output_feature).detach().cpu()
            distance = output_feature_norm @ feature_bank.t()
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.k)
            score_near = score_bank[idx_near].permute(0, 2, 1)
            feature_bank[index] = output_feature_norm.detach()
            score_bank[index] = output_proba1.detach().cpu()
        if args.shot > 0:
            score_bank[few_short_index] = label_bank[few_short_index]

        KNN_loss = -torch.mean(torch.log(torch.bmm(output_proba1.unsqueeze(1), score_near.cuda())).sum(-1))
        loss = KNN_loss
        output_proba_mean = output_proba1.mean(dim=0)
        KL_loss = torch.sum(output_proba_mean * torch.log(output_proba_mean + 1e-5))
        loss += KL_loss
        Dcls_loss = -torch.log(torch.bmm(nn.Softmax(dim=1)(out).unsqueeze(1), labels.unsqueeze(2))).mean()
        loss += Dcls_loss*10
        if global_step % 50 == 0:
            print('DC loss:', Dcls_loss.item(), 'ACC of domain classifier: ', prob)

        if args.MixMatch and flag == 1:
            # labeled_ID = []
            # unlabeled_ID = []
            # for i in range(len(index)):
            #     if index[i] in few_short_index:
            #         labeled_ID.append(i)
            #     else:
            #         unlabeled_ID.append(i)
            # num_labeled = len(labeled_ID)
            # num_unlabeled = len(unlabeled_ID)
            # if num_labeled > 0:
            # print(index_labeled)
            with torch.no_grad():
                y_labeled = label_bank[index_labeled]
                # input_X1_cuda = torch.cat([input_X.cuda(), input_X1_cuda, input_X2.cuda()], dim=0)
                x2 = model.freeze_netF(input_X2.cuda())
                dls_out2 = model.netD(x2, args, netG, test=True)
                output_feature2, _ = model.netB(model.netF(input_X2.cuda()), s=args.SMAX, t=t, d_cls_out=dls_out2, type=args.type)
                output_proba2 = nn.Softmax(dim=1)(model.netC(output_feature2)).detach()

                proba_temperature = ((output_proba1[len(input_X):] + output_proba2) / 2) ** (1 / 0.5)
                proba_unlabeled = proba_temperature / proba_temperature.sum(dim=1, keepdim=True)
                # proba_mask = ((dls_out[len(input_X):] + dls_out2) / 2) ** (1 / 0.5)
                # proba_masks = proba_mask / proba_mask.sum(dim=1, keepdim=True)
                l = np.random.beta(0.75, 0.75)
                l = max(l, 1 - l)
                input_X1_cuda = torch.cat([input_X1_cuda, input_X2.cuda()], dim=0)
                input_y = torch.cat([y_labeled.cuda(), proba_unlabeled, proba_unlabeled], dim=0)
                # mask_label = torch.zeros(t+1)
                # mask_label[t] = 1
                # mask_label = mask_label.repeat(len(input_X),1).cuda()
                # mask = torch.cat([mask_label,dls_out[len(input_X):], dls_out2], dim=0)
                # print(mask.shape)
                # input_X1_cuda = torch.cat([input_X2.cuda(), input_X1_cuda], dim=0)
                # input_y = torch.cat([proba_unlabeled, proba_unlabeled], dim=0)
                random_index = torch.randperm(input_X1_cuda.size(0))
                input_X1_cuda = input_X1_cuda.cpu()
                mixed_X = l * input_X1_cuda + (1 - l) * input_X1_cuda[random_index]
                mixed_y = l * input_y + (1 - l) * input_y[random_index]
                # mixed_mask = l * mask + (1 - l) * mask[random_index]
                x_mix = model.freeze_netF(mixed_X.cuda())
                dls_out_mix = model.netD(x_mix, args, netG, test=True)
            output_feature_mix, _ = model.netB(model.netF(mixed_X.cuda()), s=args.SMAX, t=t, d_cls_out=dls_out_mix, type=args.type)
            output_proba_mix = nn.Softmax(dim=1)(model.netC(output_feature_mix))
            lambda_u = np.clip(global_step / t_total, 0.0, 1.0) * 100
            MixMatch_loss = -torch.log(
                torch.bmm(output_proba_mix[:len(input_X)].unsqueeze(1), mixed_y[:len(input_X)].unsqueeze(2))).mean() \
                            + lambda_u * torch.mean((mixed_y[len(input_X):] - output_proba_mix[len(input_X):]) ** 2)
            # Dclsmix_loss = nn.CrossEntropyLoss()(out_mix,labels_mix)
            loss += MixMatch_loss

        loss /= args.grad_iter

        loss.backward()

        for n, p in model.bottle.named_parameters():
            if n.find('bias') == -1:
                mask_ = ((1 - masks_old)).view(-1, 1).expand(768, 768).cuda()
                p.grad.data *= mask_
            else:
                mask_ = ((1 - masks_old)).squeeze().cuda()
                p.grad.data *= mask_

        for n, p in model.head.named_parameters():
            if n.find('weight_v') != -1:
                masks__ = masks_old.view(1, -1).expand(args.class_num, 768)
                mask_ = ((1 - masks__)).cuda()
                p.grad.data *= mask_

        for n, p in model.normalazation.named_parameters():
            mask_ = ((1 - masks_old)).view(-1).cuda()
            p.grad.data *= mask_

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
        with torch.no_grad():
            feature = model.freeze_netF(input_X1.cuda())
            feature = feature.detach().view(-1, 3, 16, 16).cuda()
        results = generator.forward(feature)
        generator_loss = generator.loss_function(*results, M_N=0.00025)
        loss_g = generator_loss['loss']
        loss_g /= args.grad_iter
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 10000)

        if (global_step + 1) % args.grad_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            optimizer_g.step()
            optimizer_g.zero_grad()
            scheduler_g.step()

    torch.save(model.state_dict(), osp.join(args.model_dir, "source_mode{}_{}.pt".format(args.mode, t)))
    torch.save(generator.state_dict(), osp.join(args.model_dir, "generator_mode{}_{}.pt".format(args.mode, t)))
    return model, ACC_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation on office-home dataset')
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--worker', type=int, default=8, help="dataloader's number of workers")
    parser.add_argument('--max_epoch', type=int, default=10, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('--grad_iter', type=int, default=2, help="batch_size")
    parser.add_argument('--t_total', type=int, default=500, help="batch_size")
    parser.add_argument('--k', type=int, default=3, help="number of neighborhoods")
    parser.add_argument('--dset', type=str, default='office31', help='office31,officeHome,digit')
    parser.add_argument('--order', type=str, default='a2d', help='training order')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=768)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--log_file', type=str, default='SFDA')
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--labeled_num', type=int, default=16)
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--pseudo_sample', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0:baseline,SMAX=100;1:baseline,SMAX=10;2:MixMatch,SMAX=10')
    parser.add_argument('--type', type=str, default='soft_mask', help='hard_mask or soft_mask')
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train(default: 200)')
    parser.add_argument('--test_every', type=int, default=10, metavar='N', help='test after every epochs')
    parser.add_argument('--z_dim', type=int, default=62, metavar='N', help='the dim of latent variable z(default: 20)')
    parser.add_argument('--input_dim', type=int, default=768, metavar='N', help='input dim(default: 28*28 for MNIST)')
    parser.add_argument('--input_channel', type=int, default=3, metavar='N', help='input channel(default: 1 for MNIST)')
    parser.add_argument('--lrf', type=float, default=0.01)
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./VIT_weight/vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # init setting of each dataset
    if args.dset == 'office31':
        args.order = 'a2d'
        args.z_dim = 62
        args.class_num = 31
        args.img_size = 256
        dset_loaders = office31_load(args)
    if args.dset == 'office-home':
        # args.order = 'A2C2P2R'
        args.z_dim = 130
        args.class_num = 65
        args.img_size = 256
        dset_loaders = officeHome_load(args)
    if args.dset == 'visda':
        args.order = 't2v'
        args.z_dim = 130
        args.class_num = 12
        args.img_size = 256
        dset_loaders = visda_load(args)
    args.task_num = len(args.order.split('2'))

    if args.mode == 0:
        args.MixMatch = False
        args.SMAX = 100
        args.freeze_layer = 4
    elif args.mode == 1:
        args.MixMatch = True
        args.SMAX = 100
        args.freeze_layer = 4

    current_folder = "./"
    args.output_dir = osp.join(current_folder, 'SS_OUTPUT_mixmodel_cda927', args.dset, args.order)
    args.model_dir = osp.join(args.output_dir, 'model')
    # model_path = args.model_dir + '/source_{}.pt'.format(0)
    model_path_final = args.model_dir + '/source_final_{}.pt'.format(10)
    cvae_path = args.model_dir + '/generator_0.pt'
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not osp.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not osp.exists(model_path_final):
        args.out_file = open(osp.join(args.output_dir, args.log_file + '_mode_{}.txt'.format(args.mode)), 'a+')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        for t in range(args.task_num):
            args.out_file.write('Training task {}'.format(t) + '\n')
            args.out_file.flush()
            model_path = args.model_dir + '/source_{}.pt'.format(t)
            cvae_path = args.model_dir + '/generator_{}.pt'.format(t)
            if t == 0:
                if not osp.exists(model_path):
                    # Train source model
                    train_source_g(args, dset_loaders)
                    # exit()
                config = CONFIGS[args.model_type]
                model = VisionTransformer_DomainClassifier_cda(config, args.img_size, zero_head=True, num_classes=args.class_num,
                                                               bottleneck=args.bottleneck, freeze_layers=args.freeze_layer, domain_num=2).cuda()
                # model = torch.nn.DataParallel(model)
                # model = model.cuda()
                weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
                # del unused weight
                # for k in ['domain_classifier.4.weight', 'domain_classifier.4.bias']:
                #     del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)
                generator = {}
                gen = VAE().cuda()
                gen.load_state_dict(torch.load(cvae_path), strict=False)
                generator[0] = gen
                ACC_list = []
            else:
                _, ACC_list = train_target_near(args, t, model, dset_loaders, netG=generator, ACC_list=ACC_list)
                model_path = args.model_dir + '/source_mode{}_{}.pt'.format(args.mode, t)
                config = CONFIGS[args.model_type]
                model = VisionTransformer_DomainClassifier_cda(config, args.img_size, zero_head=True, num_classes=args.class_num,
                                                               bottleneck=args.bottleneck, freeze_layers=args.freeze_layer,
                                                               domain_num=t + 2).cuda()
                # model = torch.nn.DataParallel(model)
                # model = model.cuda()
                weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda())
                # del unused weight
                for k in ['domain_classifier.4.weight', 'domain_classifier.4.bias']:
                    del weights_dict[k]
                model.load_state_dict(weights_dict, strict=False)
                generator = {}
                for i in range(t + 1):
                    if i == 0:
                        cvae_path = args.model_dir + '/generator_{}.pt'.format(i)
                    else:
                        cvae_path = args.model_dir + '/generator_mode{}_{}.pt'.format(args.mode, i)
                    gen = VAE().cuda()
                    gen.load_state_dict(torch.load(cvae_path), strict=False)
                    generator[i] = gen

        torch.save(model.state_dict(), osp.join(args.model_dir, "source_F_final_{}.pt".format(args.mode)))
        joblib.dump(ACC_list, args.output_dir + '/ACC_list_mode_{}.joblib'.format(args.mode))
