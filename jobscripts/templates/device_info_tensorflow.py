import tensorflow as tf

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print('Is there a GPU?                   .............. Yes')
else:
    print('Is there a GPU?                   .............. No')

# Get number of GPUs
print('How many GPUs do we have          .............. ',len(gpus))

# Get GPU properties
if len(gpus) > 0:
    device_properties = tf.config.experimental.get_device_details(gpus[0])
    print('GPU properties                    .............. ')
    for key, value in device_properties.items():
        print(f'    {key}: {value}')
else:
    print('GPU properties                    .............. Not available')

# Get supported GPU micro-architectures
print('Supported GPU micro-architectures .............. ',tf.config.experimental.list_physical_devices('GPU'))

# Get current GPU micro-architecture
#if len(gpus) > 0:
#    major, minor = tf.config.experimental.get_device_capability(gpus[0])
#    print('Which GPU micro-architecture is this?........... ',f'{major}.{minor}')
#else:
#    print('Which GPU micro-architecture is this?........... Not available')

# Number of threads available on host
# print('Number of threads available on host ............ ',tf.config.threading.intra_op_parallelism_threads())
print('Number of threads available on host ............ ', tf.config.threading.get_intra_op_parallelism_threads() )

