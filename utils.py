import os
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os.path as osp
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import network
import torch.optim as optim


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.mean(entropy, dim=1)
    return entropy

def discrepancy(out1, out2):
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    #return (F.kl_div(F.log_softmax(out1), out2_t) + F.kl_div(F.log_softmax(out2), out1_t)) / 2
    #return F.kl_div(F.log_softmax(out1), out2_t, reduction='none')
    return (F.kl_div(F.log_softmax(out1), out2_t, reduction='none')
    +F.kl_div(F.log_softmax(out2), out1_t, reduction='none')) / 2


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self,
                 num_classes,
                 epsilon=0.1,
                 use_gpu=True,
                 size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs=netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def cal_acc_proto(loader, netF, netC,proto):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netF.forward(inputs)  # a^t
            #outputs=F.normalize(outputs,dim=-1,p=2)
            #outputs = netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
        all_output_np=np.array(all_output)
        center=proto
        center = center.float().detach().cpu().numpy()
        dist=torch.from_numpy(cdist(all_output_np,center))
        _, predict = torch.min(dist, 1)
        accuracy = torch.sum(
            torch.squeeze(predict).float() == all_label).item() / float(
                all_label.size()[0])
    return accuracy, accuracy


def cal_acc_sda(loader, netF,netC,t=0,s=100):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,_ = netF.forward(inputs,t=t)  # a^t
            outputs = netC(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent

def cal_acc_visda(loader, netF, netC,t=0, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netF(inputs,t=t)[0])
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent


def cal_acc_vit(loader, model,t=0,s=100,flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs,_ = model.forward_features(inputs,t=t)  # a^t
            outputs = model.fea_classifier(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    mean_ent = torch.mean(Entropy(
        nn.Softmax(dim=1)(all_output))).cpu().data.item()
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy, mean_ent

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def test_model(model, sample_size, path, verbose=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(
        model.sample(sample_size).data,
        path + '.jpg',
        nrow=6,
    )
    if verbose:
        print('=> generated sample images at "{}".'.format(path))

def inversesigmoid(x):
    bias = torch.ones(x.shape)
    return -torch.log(1/(x+1e-8)-1)

def cal_acc_clf(loader,model,d_cls,label=0):
    d_cls.eval()
    model.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            labels = (torch.ones(inputs.shape[0])*label).long()
            outputs = model.netF(inputs)
            outputs=d_cls(outputs)
            # _,pred=torch.max(output,1)
            # accuracy = torch.sum(
            #     torch.squeeze(pred).float() == labels).item() / float(
            #         labels.size()[0])
            # print(loss,accuracy)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy( nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy

def cal_acc_domain_agnostic(loader,netF,netF_final,netC,d_cls,s=100):
    netF.eval()
    netF_final.eval()
    netC.eval()
    d_cls.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netF.forward(inputs,t=0,generate=True)
            domain_pre = d_cls(features)
            _, pre = torch.max(domain_pre, 1)
            if pre.shape[0] == 1:
                continue
            mask = torch.squeeze(pre).int()
            for k in range(mask.shape[0]):
                input = inputs[k,:]
                input = input[np.newaxis,:]
                outputs,_ = netF_final.forward(input,t=mask[k],s=s)  # a^t
                outputs = netC(outputs)
                if k==0:
                    batch_outputs = outputs
                else:
                    batch_outputs = torch.cat((batch_outputs, outputs))
            # print(batch_outputs.shape)
            if start_test:
                all_output = batch_outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, batch_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])
    # mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy

def generate_feature(args,dset_loaders,t,netF):
    # Step 1: 载入数据
    # mnist_test, mnist_train, classes = dataloader(args.batch_size, args.num_worker)
    train_dataset = dset_loaders[str(t) +'gr']
    test_dataset = dset_loaders[str(t) +'gr']

    # 查看每一个batch图片的规模
    x, label, _ = iter(train_dataset).__next__()  # 取出第一批(batch)训练所用的数据集
    print(' img : ', x.shape)  # img :  torch.Size([batch_size, 1, 28, 28])， 每次迭代获取batch_size张图片，每张图大小为(1,28,28)

    # Step 2: 准备工作 : 搭建计算流程
    model = network.CVAE(input_dim=args.input_dim,y_dim=args.class_num,z_dim=args.z_dim).cuda()  # 生成AE模型，并转移到GPU上去
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
            outputs = netF.forward(x,t=t,generate=True)
            # print(outputs.shape)
            # a,pr = torch.min(outputs,1)
            # print(a)
            # bias = (torch.ones(outputs.shape)*10).long().cuda()
            # print(outputs.shape)  # [64,512]
            # data =torch.tanh(outputs)
            data = outputs
            # print(data)
            # print(data.shape)
            data = data.cuda()
            # 前向传播
            x_hat, mu, log_var = model(data, label)  # 模型的输出，在这里会自动调用model中的forward函数
            #训练样本展平，在每个样本后面连接标签的one-hot向量
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

            # if batch_index == 0:
            #     # visualize reconstructed result at the beginning of each epoch
            #     sample = x_hat.cpu()
            #     generated_image = sample[:, 0:sample.shape[1]-10].cuda()
            #     x_concat = torch.cat([x.view(-1, 1, 28, 28), generated_image.view(-1, 1, 28, 28)], dim=3)
            #     save_image(x_concat, './%s/reconstructed-%d.png' % (args.result_dir, epoch + 1))

        # 把这一个epoch的每一个样本的平均损失存起来
        loss_epoch.append(np.sum(loss_batch) / len(train_dataset.dataset))  # len(mnist_train.dataset)为样本个数

        # 测试模型
        if (epoch + 1) % args.test_every == 0:
            test_loss = cvae_test(model, test_dataset, netF,t)
            logstr = ('Epoch [{}/{}],Test-loss= {:.4f}'.format(epoch + 1, args.epochs, test_loss))
            args.out_file.write(logstr + '\n')
            args.out_file.flush()
            print(logstr)
    
    store_CVAE = model.state_dict()
    torch.save(store_CVAE, osp.join(args.model_dir, "CVAE_{}.pt".format(t)))
    return loss_epoch


'''
Domain Classifier
'''

def domain_classifier_real(args,dset_loaders,model):
    domain_num = 2

    d_cls=network.CLS_D(domain_num=2).cuda()
    optim=torch.optim.SGD(d_cls.parameters(),lr=0.01)
    d_cls.eval()
    model.eval()
    iter_s = iter(dset_loaders['0gr'])
    iter_t = iter(dset_loaders['1gr'])

    input_s,label1,_ = iter_s.next()
    if input_s.shape[0]<args.batch_size:
        input_s,label1,_ = iter_s.next()
    input_t,label2,_ = iter_t.next()
    if input_t.shape[0]<args.batch_size:
        input_t,label2,_ = iter_t.next()

    data1 = input_s.clone()
    data2 = input_t.clone()
    data1 = data1.cuda()
    data2 = data2.cuda()
    ## 加载目标域特征
    with torch.no_grad():
        feature1 = model.netF(data1)
        domain_label1 = torch.zeros(feature1.shape[0]).long().cuda()
        feature2 = model.netF(data2)
        domain_label2 = torch.ones(feature2.shape[0]).long().cuda()

    feature_data = torch.cat((feature1,feature2))
    domain_label = torch.cat((domain_label1, domain_label2))
    
    for k in range(200):
        # print(feature_data.shape)
        inputs_np=feature_data.cpu().numpy()
        labels_np=domain_label.cpu().numpy()
        state = np.random.get_state()
        np.random.shuffle(inputs_np)
        np.random.set_state(state)
        np.random.shuffle(labels_np)
    
        inputs=torch.from_numpy(inputs_np).cuda()
        labels=torch.from_numpy(labels_np).cuda().long()
        # print(inputs.shape)
        # print(labels)
        d_cls.train()
        # print(inputs.shape)
        output=d_cls(inputs)
        loss=nn.CrossEntropyLoss()(output,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if (k+1)%20 == 0:
            d_cls.eval()
            acc1 = cal_acc_clf(dset_loaders['0gr'],model, d_cls)
            # acc1 = 0
            acc2 = cal_acc_clf(dset_loaders['1gr'],model, d_cls, label=1)
            log_str = 'Task: {}, Iter:{}/{};Loss = {:.2f}, Accuracy = {:.2f}% | {:.2f}%  '.format(
                'pseudo data', k + 1, 200, loss, acc1 * 100,acc2 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
            d_cls.train()
    store_cls = d_cls.state_dict()
    torch.save(store_cls, osp.join(args.model_dir, "D_cls_save_batch.pt"))    


def domain_classifier_all(args,dset_loaders,netF):
    domain_num = 2

    d_cls=network.CLS_D(domain_num=2).cuda()
    optim=torch.optim.SGD(d_cls.parameters(),lr=0.01)
    d_cls.eval()
    netF.eval()
    
    for k in range(1000):
        iter_s = iter(dset_loaders['0gr'])
        iter_t = iter(dset_loaders['1gr'])
        input_s,label1,_ = iter_s.next()
        if input_s.shape[0]<args.batch_size:
            input_s,label1,_ = iter_s.next()
        input_t,label2,_ = iter_t.next()
        if input_t.shape[0]<args.batch_size:
            input_t,label2,_ = iter_t.next()

        data1 = input_s.cuda()
        data2 = input_t.cuda()
        ## 加载目标域特征
        with torch.no_grad():
            feature1 = netF.forward(data1,t=0,generate=True)
            domain_label1 = torch.zeros(feature1.shape[0]).long().cuda()
            feature2 = netF.forward(data2,t=1,generate=True)
            domain_label2 = torch.ones(feature2.shape[0]).long().cuda()

        feature_data = torch.cat((feature1,feature2))
        domain_label = torch.cat((domain_label1, domain_label2))
        # print(feature_data.shape)
        inputs_np=feature_data.cpu().numpy()
        labels_np=domain_label.cpu().numpy()
        state = np.random.get_state()
        np.random.shuffle(inputs_np)
        np.random.set_state(state)
        np.random.shuffle(labels_np)
    
        inputs=torch.from_numpy(inputs_np[:64]).cuda()
        labels=torch.from_numpy(labels_np[:64]).cuda().long()
        # print(inputs.shape)
        # print(labels)
        d_cls.train()
        # print(inputs.shape)
        output=d_cls(inputs)
        loss=nn.CrossEntropyLoss()(output,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        d_cls.eval()
        acc1 = cal_acc_clf(dset_loaders['0gr'],netF, d_cls)
        # acc1 = 0
        acc2 = cal_acc_clf(dset_loaders['1gr'],netF, d_cls, label=1)
        log_str = 'Task: {}, Iter:{}/{};Loss = {:.2f}, Accuracy = {:.2f}% | {:.2f}%  '.format(
            'pseudo data', k + 1, 1000, loss, acc1 * 100,acc2 * 100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str)

        d_cls.train()
    store_cls = d_cls.state_dict()
    torch.save(store_cls, osp.join(args.model_dir, "D_cls_save_all.pt"))    

def domain_classifier_cvae(args,dset_loaders,model,netG=None):
    domain_num = 2

    d_cls=network.CLS_D(domain_num=2).cuda()
    optim=torch.optim.SGD(d_cls.parameters(),lr=0.01)
    d_cls.eval()
    model.eval()
    
    iter_data = {}
    ## 加载目标域特征
    for k in range(200):
        if netG ==None:
            for i in range(domain_num):
                iter_data[str(i)] = iter(dset_loaders[str(i) +'gr'])
        else:
            # print('true')
            for i in range(domain_num):
                with torch.no_grad():
                    # 随机从隐变量的分布中取隐变量
                    z = torch.randn(args.batch_size, args.z_dim).cuda()  # 每一行是一个隐变量，总共有batch_size行
                    c = np.zeros(shape=(z.shape[0],))
                    # rand = np.random.randint(0, 31)
                    # rand = i
                    # print(f"Random number: {rand}")
                    for m in range(z.shape[0]):
                        c[m] = np.random.randint(0, args.class_num) 
                    c = torch.FloatTensor(c)
                    # print(c)
                    # 对隐变量重构
                    random_res = netG[str(i)].decode(z, c).cpu()
                    #模型的输出矩阵：每一行的末尾都加了one-hot向量，要去掉这个one-hot向量
                    generated_image = random_res[:, 0:random_res.shape[1]-args.class_num]
                    # bias = torch.ones(generated_image.shape).long()
                    gen = generated_image 
                    # gen = inversesigmoid(generated_image)
                    # print(gen.shape)
                    # print(gen)
                iter_data[str(i)] = gen
        with torch.no_grad():
            for t in range(domain_num):
                # if t==0:
                #     continue
                if netG==None:
                    inputs, labels, _ = iter_data[str(t)].next()
                    if inputs.shape[0] < args.batch_size:
                        inputs, labels, _ = iter_data[str(t)].next()
                    inputs = inputs.cuda()
                    features = model.netF(inputs)
                else:
                    features = iter_data[str(t)]
                if t==0:
                    feature_data = features
                    domain_label = torch.zeros(features.shape[0]).long().cuda()
                else:
                    feature_data = torch.cat((feature_data,features))
                    domain_label = torch.cat((domain_label, (torch.ones(features.shape[0])).long().cuda()))
            # print(feature_data.shape)
            inputs_np=feature_data.cpu().numpy()
            labels_np=domain_label.cpu().numpy()
            state = np.random.get_state()
            np.random.shuffle(inputs_np)
            np.random.set_state(state)
            np.random.shuffle(labels_np)
        
            inputs=torch.from_numpy(inputs_np).cuda()
            labels=torch.from_numpy(labels_np).cuda().long()
        # print(inputs.shape)
        # print(labels)
        d_cls.train()
        # print(inputs.shape)
        output=d_cls(inputs)
        loss=nn.CrossEntropyLoss()(output,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (k+1)%20 == 0:
            d_cls.eval()
            acc1 = cal_acc_clf(dset_loaders['0gr'],model, d_cls)
            # acc1 = 0
            acc2 = cal_acc_clf(dset_loaders['1gr'],model, d_cls, label=1)
            log_str = 'Task: {}, Iter:{}/{};Loss = {:.2f}, Accuracy = {:.2f}% | {:.2f}%  '.format(
                'pseudo data', k + 1, 200, loss, acc1 * 100,acc2 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)

            d_cls.train()
    store_cls = d_cls.state_dict()
    torch.save(store_cls, osp.join(args.model_dir, "D_cls_cvae.pt"))       


def domain_classifier_net(args,dset_loaders,model,features,labels):
    domain_num = 2

    d_cls=network.CLS_D(domain_num=2).cuda()
    optim=torch.optim.SGD(d_cls.parameters(),lr=0.01)
    d_cls.eval()
    
    for k in range(200):
        # print(feature_data.shape)
        inputs_np=features.cpu().numpy()
        labels_np=labels.cpu().numpy()
        state = np.random.get_state()
        np.random.shuffle(inputs_np)
        np.random.set_state(state)
        np.random.shuffle(labels_np)
    
        inputs=torch.from_numpy(inputs_np[:64]).cuda()
        labels=torch.from_numpy(labels_np[:64]).cuda().long()
        # print(inputs.shape)
        # print(labels)
        d_cls.train()
        # print(inputs.shape)
        output=d_cls(inputs)
        loss=nn.CrossEntropyLoss()(output,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (k+1)%20 == 0:
            d_cls.eval()
            acc1 = cal_acc_clf(dset_loaders['0gr'],model, d_cls)
            # acc1 = 0
            acc2 = cal_acc_clf(dset_loaders['1gr'],model, d_cls, label=1)
            log_str = 'Task: {}, Iter:{}/{};Loss = {:.2f}, Accuracy = {:.2f}% | {:.2f}%  '.format(
                'pseudo data', k + 1, 200, loss, acc1 * 100,acc2 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)

            d_cls.train()
    store_cls = d_cls.state_dict()
    torch.save(store_cls, osp.join(args.model_dir, "D_cls_net_base.pt"))          

def domain_classifier_wgan(args,dset_loaders,model,netG):
    domain_num = 2

    d_cls=network.CLS_D(domain_num=2).cuda()
    optim=torch.optim.SGD(d_cls.parameters(),lr=0.01)
    d_cls.eval()
    model.eval()
    
    iter_data = {}
    ## 加载目标域特征
    for k in range(200):
        for i in range(domain_num):
            with torch.no_grad():
                features = netG[str(i)].sample(32).data
                features = features.squeeze()
            iter_data[str(i)] = features
        with torch.no_grad():
            for t in range(domain_num):
                features = iter_data[str(t)]
                if t==0:
                    feature_data = features
                    domain_label = torch.zeros(features.shape[0]).long().cuda()
                else:
                    feature_data = torch.cat((feature_data,features))
                    domain_label = torch.cat((domain_label, (torch.ones(features.shape[0])).long().cuda()))
            # print(feature_data.shape)
            inputs_np=feature_data.cpu().numpy()
            labels_np=domain_label.cpu().numpy()
            state = np.random.get_state()
            np.random.shuffle(inputs_np)
            np.random.set_state(state)
            np.random.shuffle(labels_np)
        
            inputs=torch.from_numpy(inputs_np[:64]).cuda()
            labels=torch.from_numpy(labels_np[:64]).cuda().long()
        # print(inputs.shape)
        # print(labels)
        d_cls.train()
        # print(inputs.shape)
        output=d_cls(inputs)
        loss=nn.CrossEntropyLoss()(output,labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (k+1)%20 == 0:
            d_cls.eval()
            acc1 = cal_acc_clf(dset_loaders['0gr'],model, d_cls)
            # acc1 = 0
            acc2 = cal_acc_clf(dset_loaders['1gr'],model, d_cls, label=1)
            log_str = 'Task: {}, Iter:{}/{};Loss = {:.2f}, Accuracy = {:.2f}% | {:.2f}%  '.format(
                'pseudo data', k + 1, 200, loss, acc1 * 100,acc2 * 100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)

            d_cls.train()
    store_cls = d_cls.state_dict()
    torch.save(store_cls, osp.join(args.model_dir, "D_cls_wgan_base.pt"))       





