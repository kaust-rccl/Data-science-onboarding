import os
import torch
import torch.distributions as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed Synchronous SGD Example using CPU """
    # Create a model on the mps device
    mps_device = torch.device("cpu")
    # define the model
    model = torch.nn.Linear(10, 1)
    model.to(mps_device)
    # define the loss function
    criterion = torch.nn.MSELoss()
    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # define the data
   # Create a Tensor directly on the mps device
    x = torch.randn(size, 10, device=mps_device)
    y = torch.randn(size, 1, device=mps_device)
    # define the number of iterations
    for i in range(10):
        # define the data for each process
        local_data = x[rank:size:size]
        local_target = y[rank:size:size]

        # define the data for each process
        local_data = local_data.clone().detach().requires_grad_(True)
        local_target = local_target.clone().detach().requires_grad_(True)

        # define the data for each process
        local_data = local_data.to('cpu')
        local_target = local_target.to('cpu')

        # define the data for each process
        local_data = torch.autograd.Variable(local_data)
        local_target = torch.autograd.Variable(local_target)

        # define the data for each process

        optimizer.zero_grad()
        # define the data for each process

        output = model(local_data)

        # define the data for each process
        loss = criterion(output, local_target)

        # define the data for each process
        loss.backward()

        # define the data for each process
        optimizer.step()

        # define the data for each process
        print('Process: {}, Iteration: {}, Loss: {}'.format(rank, i, loss.item()))

    pass

def init_processes(rank, size, fn, backend='gloo'):
    # """ initialize the distributed environment. """
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group(backend, rank=rank, world_size=size)
    # fn(rank, size)
    dist_url = "env://" # default
        # only works with torch.distributed.launch // torch.run

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print (f"{local_rank} : {world_size} , {rank} \n")
    dist.init_process_group(backend, 
        rank=rank, 
        world_size=world_size, 
        init_method=dist_url)
        # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    fn(rank, size)


if __name__ == '__main__':
    size = 2
    processes = []
    for rank in range(size):
        p = mp.Process(target=run, args=(rank, size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('done')