import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from dataload.dataload import DataLoad
from loss.loss import Loss, MyLoss, UcertRELossL1
from metric.metric import Metric
from summary.summary import Summary
from config import args
from model.mobileunetv2 import mobileunetv2
from model.ours import MDCnet, AIRnet
import utility
from tqdm import tqdm
from loss.uncertrainty import Ucertl2MaskedMSELoss, MaskedMSELoss
from torch import nn

import datetime


def train(args, gpu=0):
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M/")

    # torch.cuda.set_device(gpu)

    ### Load data
    data_train = DataLoad(args, 'train')
    data_val = DataLoad(args, 'val')

    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(
        data_val, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=args.num_summary, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_val,
        drop_last=False)

    ### Set up Network
    # net = mobileunetv2(args)
    net = MDCnet()
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        if gpu == 0:
            net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()}, strict=False)
            print('Load network parameters from : {}'.format(args.pretrain))

    ### Set up Loss
    # loss = Loss(args)
    loss = MyLoss(args)
    val_Loss = Loss(args)

    ### Set up Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

    ### Set up Metric
    metric = Metric(args)

    if gpu == 0:
        # utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

        writer_train = Summary(args.save_dir, 'train', args,loss.loss_name, metric.metric_name)
        writer_val = Summary(args.save_dir, 'val', args, val_Loss.loss_name, metric.metric_name)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    ### Training
    for epoch in range(args.epochs+1):
        net.train()
        sampler_train.set_epoch(epoch)

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            
            if epoch == 0 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)
            # pred_2, pred_3, pred_4, s_2, s_3, s_4 = net(sample)

            sample['epoch'] = epoch

            loss_sum, loss_val = loss(sample, output)
            # Divide by batch size
            # loss_sum = loss_sum / loader_train.batch_size
            # loss_val = loss_val / loader_train.batch_size

            loss_sum.backward()

            optimizer.step()

            if gpu == 0:
                # metric_val = metric.evaluate(sample['gt'], output)
                metric_val = metric.evaluate(sample['gt'], output[3])
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss1 = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str, list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)
        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, (output[3], output[-1]))
            # writer_train.update(epoch, sample, output)
            
            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.state_dict(),
                    'args': args
                }
            torch.save(state, f'{args.save_dir}/model_{epoch:05d}.pt')
        while(not os.path.exists('{}/model_{:05d}.pt'.format(args.save_dir, epoch))):
            #print('waiting for model')
            pass
        if gpu == 0:
            checkpoint = torch.load('{}/model_{:05d}.pt'.format(args.save_dir, epoch))
            net.load_state_dict(checkpoint['net'], strict=False)
        
        ### Validation
        torch.set_grad_enabled(False)
        net.eval()
        num_sample = len(loader_val) * loader_val.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            output = net(sample)
            # loss_sum, loss_val = val_Loss(sample, output[0])
            loss_sum, loss_val = val_Loss(sample, output[0])

            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size
            if gpu == 0:
                # metric_val = metric.evaluate(sample['gt'], output[0])
                metric_val = metric.evaluate(sample['gt'], output[0])
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f} | epoch {}'.format(
                    'Val', current_time, log_loss / log_cnt, epoch)
                pbar.set_description(error_str)
                #print(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()
            # writer_val.update(epoch, sample, output[0])
            writer_val.update(epoch, sample, output)
            # writer_val.save(epoch, batch, sample, output[0])
            writer_val.save(epoch, batch, sample, output)

        torch.set_grad_enabled(True)
        scheduler.step()


# --train_step2
# --pretrain pretrained/model_00099.pt
def train_step2(args, gpu=0):
    timestamp = datetime.datetime.now().strftime("%m-%d-%H.%M/")

    torch.cuda.set_device(gpu)

    ### Load data
    data_train = DataLoad(args, 'train')
    data_val = DataLoad(args, 'val')

    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(
        data_val, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=args.num_summary, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_val,
        drop_last=False)

    base_net = MDCnet()
    base_net.cuda()

    assert args.pretrain is not None, 'pretrain is None'

    checkpoint = torch.load(args.pretrain)
    base_net.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['net'].items()}, strict=False)

    base_net.eval()

    model = AIRnet()
    model.cuda()

    if args.pretrain_step2 is not None:
        assert os.path.exists(args.pretrain_step2), \
            "file not found: {}".format(args.pretrain_step2)

        checkpoint = torch.load(args.pretrain_step2)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()}, strict=False)
        # net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'], strict=True)

    optimizer, scheduler = utility.make_optimizer_scheduler(args, model)

    ### Set up Metric
    metric = Metric(args)

    loss = UcertRELossL1()
    val_Loss = Loss(args)

    if gpu == 0:
        # utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

        writer_train = Summary(args.save_dir, 'train', args, loss.loss_name, metric.metric_name)
        writer_val = Summary(args.save_dir, 'val', args, val_Loss.loss_name, metric.metric_name)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train) + 1.0

    for epoch in range(args.epochs + 1):
        model.train()
        sampler_train.set_epoch(epoch)

        num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}

            if epoch == 0 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            with torch.no_grad():
                pred_base, s_4 = base_net(sample)

            res, pred = model(pred_base, sample)

            gt = sample['gt']
            loss_sum, loss_val = loss(pred_base, res, s_4, gt)

            loss_sum.backward()
            optimizer.step()

            if gpu == 0:
                # metric_val = metric.evaluate(sample['gt'], output)
                metric_val = metric.evaluate(sample['gt'], pred)
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss1 = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str, list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_train.update_step2(epoch, sample, (pred_base, s_4), (res, pred))
            # writer_train.update(epoch, sample, output)

            if args.save_full or epoch == args.epochs:
                state = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': model.state_dict(),
                    'args': args
                }
            torch.save(state, f'{args.save_dir}/model_step2_{epoch:05d}.pt')

        while not os.path.exists('{}/model_step2_{:05d}.pt'.format(args.save_dir, epoch)):
            # print('waiting for model')
            pass
        if gpu == 0:
            checkpoint = torch.load('{}/model_step2_{:05d}.pt'.format(args.save_dir, epoch))
            model.load_state_dict(checkpoint['net'], strict=False)

        ### Validation
        torch.set_grad_enabled(False)
        model.eval()
        num_sample = len(loader_val) * loader_val.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
            with torch.no_grad():
                pred_base, s_4 = base_net(sample)
                res, pred = model(pred_base, sample)

            loss_sum, loss_val = val_Loss(sample, pred)

            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size
            if gpu == 0:
                # metric_val = metric.evaluate(sample['gt'], output[0])
                metric_val = metric.evaluate(sample['gt'], pred)
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f} | epoch {}'.format(
                    'Val', current_time, log_loss / log_cnt, epoch)
                pbar.set_description(error_str)
                # print(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()
            # writer_val.update(epoch, sample, output[0])
            writer_val.update_step2(epoch, sample, (pred_base, s_4), (res, pred))
            # writer_val.save(epoch, batch, sample, output[0])
            writer_val.save(epoch, batch, sample, (pred_base, s_4))

        torch.set_grad_enabled(True)
        scheduler.step()


def test(args):
    ### Prepare dataset
    data_test = DataLoad(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)

    ###  Set up Network
    # net = mobileunetv2(args)
    net = MDCnet()
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        net.load_state_dict({k.replace('module.',''):v for k, v in checkpoint['net'].items()},strict=False)
        # net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'], strict=True)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    output_path = os.path.join('results/', args.test_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    flist = []
    pbar = tqdm(total=num_sample)
    import cv2
    from matplotlib import pyplot as plt

    RUN_TIME = []

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}

        start = time.time()

        output = net(sample)

        end = time.time()
        # print(end - start)
        RUN_TIME.append(end - start)

        pbar.set_description('Testing')
        pbar.update(loader_test.batch_size)

        output_image = output[0].detach().cpu().numpy()
        
        def visualize(img,name):
            plt.imshow(img,cmap='jet_r')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(name)
            plt.close()

        for i in range(output_image.shape[0]):
            cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[i, 0, :, :]).astype(np.float32))
            # visualize(output_image[i, 0, :, :],os.path.join(output_path, f'{batch}.png'))

            flist.append(f'{batch}.exr')

        with open(f'{output_path}/data.list', 'w') as f:
            for item in flist:
                f.write("%s\n" % item)

    AVE_RUNTIME = np.array(RUN_TIME).sum() / len(RUN_TIME)
    # print('AVE_RUNTIME', AVE_RUNTIME)
    pbar.close()


def test_step2(args):
    ### Prepare dataset
    data_test = DataLoad(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False)

    ###  Set up Network
    # net = mobileunetv2(args)
    base_net = MDCnet()
    base_net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        base_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()}, strict=False)
        # net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'], strict=True)

    base_net.eval()

    res_net = AIRnet()
    res_net.cuda()

    if args.pretrain_step2 is not None:
        assert os.path.exists(args.pretrain_step2), \
            "file not found: {}".format(args.pretrain_step2)

        checkpoint = torch.load(args.pretrain_step2)
        res_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()}, strict=False)
        # net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'], strict=True)

    res_net.eval()

    num_sample = len(loader_test) * loader_test.batch_size

    output_path = os.path.join('results_step2/', args.test_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    flist = []
    pbar = tqdm(total=num_sample)
    import cv2
    from matplotlib import pyplot as plt

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() if val is not None and not key == 'name' else val for key, val in sample.items()}

        start = time.time()

        pred_base, s_4 = base_net(sample)

        end1 = time.time()
        # print('run_time:{}ms'.format((end1-start)*1000))

        res, pred = res_net(pred_base, sample)

        end2 = time.time()
        # print('run_time_step2:{}ms'.format((end2-end1)*1000))
        pbar.set_description('Testing')
        pbar.update(loader_test.batch_size)

        output_image = pred.detach().cpu().numpy()

        def visualize(img, name):
            plt.imshow(img, cmap='jet_r')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(name)
            plt.close()

        for i in range(output_image.shape[0]):
            cv2.imwrite(os.path.join(output_path, f'{batch}.exr'), (output_image[i, 0, :, :]).astype(np.float32))
            # visualize(output_image[i, 0, :, :],os.path.join(output_path, f'{batch}.png'))

            flist.append(f'{batch}.exr')

        with open(f'{output_path}/data.list', 'w') as f:
            for item in flist:
                f.write("%s\n" % item)

    pbar.close()


if __name__ == "__main__":
    print("Code V0")
    current_time = time.strftime('%y%m%d_%H%M%S')
    if args.test_only:
        test(args)
    elif args.test_only_step2:
        test_step2(args)
    elif args.train_step2:
        save_dir = 'experiments_step2/' + args.save + current_time
        args.save_dir = save_dir
        train_step2(args)
    else:
        train(args)

        

