import imageio
import numpy as np
import matlab.engine
import io
import torch.nn as nn 

with matlab.engine.start_matlab() as eng:

    filename = r'C:\Users\ShenSheng\Desktop\00001x4.jpeg'
    jpeg_python = imageio.imread(filename)
    jpeg_matlab = eng.imread(filename)
    jpeg_matlab = np.array(jpeg_matlab)
    print('********************************')
    print('[JPEG file : {}]'.format(filename))
    print('matlab version 10 values:{}'.format(jpeg_matlab[0:9,0,0]))
    print('python version 10 values:{}'.format(jpeg_python[0:9,0,0]))
    print('********************************\n')

    filename = r'C:\Users\ShenSheng\Desktop\00001x4_.png'
    jpeg_python = imageio.imread(filename)
    jpeg_matlab = eng.imread(filename)
    jpeg_matlab = np.array(jpeg_matlab)
    print('********************************')
    print('[PNG file:{}]'.format(filename))
    print('matlab version 10 values:{}'.format(jpeg_matlab[0:9,0,0]))
    print('python version 10 values:{}'.format(jpeg_python[0:9,0,0]))
    print('********************************\n')

    filename = r'C:\Users\ShenSheng\Desktop\00001x4.png'
    jpeg_python = imageio.imread(filename)
    jpeg_matlab = eng.imread(filename)
    jpeg_matlab = np.array(jpeg_matlab)
    print('********************************')
    print('[PNG file:{}]'.format(filename))
    print('matlab version 10 values:{}'.format(jpeg_matlab[0:9,0,0]))
    print('python version 10 values:{}'.format(jpeg_python[0:9,0,0]))
    print('********************************\n')



