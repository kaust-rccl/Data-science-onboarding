from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import tensorboardX
import os, subprocess as sb
import math
from tqdm import tqdm
import time
from numpy.core.tests import test_einsum

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='Top level directory containing data')
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,help='number of Dataloader workers')
parser.add_argument('--node_local_storage',action='store_true', default=False,
                    help='extract dataset to local NVMe devices')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

hvd.init()
cp_time=time.time()
copy_done=-1
if hvd.local_rank() == 0:
        if args.node_local_storage is True:
            print("[%s] I am going to run copy task" %hvd.local_rank())
            os.makedirs(args.root_dir,exist_ok=True)   
            tdata=sb.run(['tar','zxf','/sw/csgv/dl/data/imagenet1K/ILSVR_2012_train.tar.gz','-C',
                          args.root_dir],stdout=sb.PIPE,stderr=sb.PIPE)
            vdata=sb.run(['tar','zxf','/sw/csgv/dl/data/imagenet1K/ILSVR_2012_val.tar.gz','-C',
                          args.root_dir],stdout=sb.PIPE,stderr=sb.PIPE)
            
            tdata_vol = sb.run(['du','-sh',args.train_dir],stdout=sb.PIPE,stderr=sb.PIPE)
            vdata_vol = sb.run(['du','-sh',args.val_dir],stdout=sb.PIPE,stderr=sb.PIPE)
            print(str(tdata_vol.stdout.decode('utf8').strip()))
            print(str(vdata_vol.stdout.decode('utf8').strip()))
            copy_done=0
copy_done=hvd.broadcast(torch.tensor(copy_done),0, name='data_copy_task')
cp_time=time.time()- cp_time
if copy_done.item() == 0: 
    if hvd.local_rank()==0:
        print("[%s,%s]: Copying data to %s took %s seconds" %(hvd.local_rank(),
                                                              hvd.rank(),args.root_dir,
                                                              cp_time))


torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None


# Horovod: limit # of CPU threads to be used per worker.
#torch.set_num_threads(4)

kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=allreduce_batch_size,
                                           sampler=train_sampler,drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=True)
					   #,persistent_workers=True)
                                           #prefetch_factor=256,
                                           

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.ToTensor(),]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler,
                                         num_workers=args.num_workers, 
                                         pin_memory=True,
                                         prefetch_factor=10)
                                         #persistent_workers=True)


# Set up standard ResNet-50 model.
model = models.resnet50()

# By default, Adasum doesn't need scaling up learning rate.
# For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          lr_scaler),
                      momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_allreduce,
    op=hvd.Adasum if args.use_adasum else hvd.Average)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

collect={'epoch':list(),
           'batch':list(),
           'data':list(),
           'host_to_dev':list(),
           'optim':list()}
def train(epoch):

    e_time=time.time()
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    
    
    
    end = time.time()
    io_time = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        io_time = time.time()- end

        adjust_learning_rate(epoch, batch_idx)
            
        if args.cuda:
            htod_time = time.time()
            data, target = data.cuda(), target.cuda()
            htod_time = time.time()- htod_time
            
        optimizer.zero_grad()
        
        # Split data into sub-batches of size batch_size
        train_time = time.time()
        for i in range(0, len(data), args.batch_size):
            data_batch = data[i:i + args.batch_size]
            target_batch = target[i:i + args.batch_size]
            output = model(data_batch)
            train_accuracy.update(accuracy(output, target_batch))
            loss = F.cross_entropy(output, target_batch)
            train_loss.update(loss)
            # Average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            loss.backward()
        train_time = time.time() - train_time
        # Gradient is applied across all ranks
        opt_time = time.time() 
        optimizer.step()
        opt_time = time.time() - opt_time
        
        collect['data'].append(io_time)
        collect['batch'].append(train_time)
        collect['host_to_dev'].append(htod_time)
        collect['optim'].append(opt_time)
        
        end=time.time()
    e_time= time.time() - e_time
    collect['epoch'].append(e_time)

    if hvd.rank() == 0:
        print("[%s]: Epoch: %s Loss: %s,  Accuracy: %s , epoch time: %s" %(hvd.rank(),
                                                           epoch,train_loss.avg.item(),
                                                           100. * train_accuracy.avg.item(),
                                                           e_time))
    
    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
    
    return collect
    
def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

def print_perf(perf_collection,metric):
    sum=0.0
    for e in range(len(perf_collection[metric])):
        sum+=perf_collection[metric][e]
    return sum

t_stamp=dict()
for epoch in range(resume_from_epoch, args.epochs):
    t_stamp=train(epoch)
    if (epoch+1 % 10) == 0:
        validate(epoch)
        save_checkpoint(epoch)

if (hvd.rank()== 0):
    print("Epoch_time:   %s \
           \nBreakdown: \
           \n  Disk I/O:  %s \
           \n  H to D time: %s \
           \n  Forward backward pass: %s \
           \n  Optimizer time: %s" %(
               sum(t_stamp['epoch']),
               sum(t_stamp['data']),
               sum(t_stamp['host_to_dev']),
               sum(t_stamp['batch']),
               sum(t_stamp['optim']) 
               )
           )
