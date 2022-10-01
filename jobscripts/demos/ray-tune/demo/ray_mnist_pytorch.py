# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class ConvNet(nn.Module): 
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

        
def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders():
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, download=True, transform=mnist_transforms
            ),
            batch_size=64,
            shuffle=True,
        )
    return train_loader, test_loader


def train_mnist(config, checkpoint_dir=None):
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    
    
        
    model = ConvNet().to(device)
    
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir,'checkpoint')
        model_state,optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(10):
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)  
       
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path= os.path.join(checkpoint_dir,'checkpoint')
            torch.save(
                (model.state_dict(),optimizer.state_dict()),
                        path
                        )    
            
        tune.report(mean_accuracy=acc)
        
def stopper(trial_id, result):
    return (result["mean_accuracy"] > 0.8) or ( (result["training_iteration"] > 6) and (result["mean_accuracy"] < 0.5) )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--use-gpu", action="store_true", default=True, 
                        help="enables training on GPUs")
    parser.add_argument("--cpus-per-trial",type=int, default=4, 
                        help="Number of CPUs to use for each trial from available pool")
    parser.add_argument("--gpus-per-trial",type=int, default=1, 
                        help="Number of GPUs to use for each trial from available pool")
    parser.add_argument("--num-samples", type=int, default=100, 
                        help="Number of samples/trials to run")
    parser.add_argument("--max-concurrent-trials", type=int, 
                        default=4, help="Maximum trials to start concurrently. \
                        Will start equal to or less trials depending on the available resources")
    parser.add_argument('--logs-dir', type=str,
                        default=os.path.join('logs',os.getenv("SLURM_JOBID")),
                        help="Create a logs directory within the experiment directory")
    parser.add_argument('--resume', action="store_true", default=False,
                        help="resume the experiment from after the last successful trial")
    args = parser.parse_args()   
    os.makedirs(args.logs_dir,exist_ok=True)
    
    ray.init(address=os.getenv('ip_head'),_redis_password=os.getenv('redis_password'))

    
    analysis = tune.run(
        train_mnist, 
        metric="mean_accuracy",
        mode="max",
        name="tune_demo",


        resources_per_trial={"cpu":args.cpus_per_trial,
                             "gpu": args.gpus_per_trial,  # set this for GPUs
                            },
        config={"lr": tune.uniform(0.01,1.0),
                "momentum": tune.uniform(0.1,0.9),
                "use_gpu": args.use_gpu,
                         },
        stop=stopper,
                 #{ "mean_accuracy": 0.8 },
        fail_fast=True,
        resume=args.resume,         

        num_samples=args.num_samples,
        max_concurrent_trials= args.max_concurrent_trials,
        
        verbose=2,
        local_dir=args.logs_dir, # relocates the log director from $HOME/ray_cluster/...
        log_to_file=True,        # writes stdout and stderr files to each trial's log directory
         
        sync_config=tune.SyncConfig(syncer=None),  # since we are using share directory for checkpoints
        keep_checkpoints_num=2,                    # keep a maximum of 5 checkpoints
        checkpoint_score_attr="mean_accuracy", # keep the (5) checkpoints with best mean_accuray in descending order
    )

    print("Best config is:", analysis.best_config)