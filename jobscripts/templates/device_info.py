import torch

print('Is there a GPU?                   .............. ',torch.cuda.is_available())
print('How many GPUs do we have          .............. ',torch.cuda.device_count())
print('GPU properties                    .............. ',torch.cuda.get_device_properties(torch.cuda.current_device()))
print('Supported GPU micro-architectures .............. ',torch.cuda.get_arch_list())
print('Which GPU micro-architecture is this?........... ',torch.cuda.get_device_capability())

print('Number of threads available on host ............ ',torch.get_num_threads())

