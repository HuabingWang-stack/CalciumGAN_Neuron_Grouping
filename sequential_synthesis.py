import subprocess
import os
import glob
import shutil
import time

datapaths = glob.glob(os.path.join('dataset\\raw_datas', '*.pkl'))
for i in range(10):
    if os.path.exists('runs\\'+str(i)):
        try:
            shutil.rmtree('runs\\'+str(i))
        except:
            pass
for i in range(10):
        try:
            shutil.rmtree('dataset\\tfrecords\\sl2048_'+str(i))
        except:
            pass
for datapath in datapaths:
    data_name = datapath.split('\\')[-1]
    print(data_name)
    output_dir = data_name.split('_')[-1][:-4]
    generate_tfrecord = 'python generate_tfrecords.py --input raw_datas\\'+data_name+' --output_dir tfrecords\\sl2048_'+output_dir+' --sequence_length 2048 --normalize'
    generate_tfrecord_process = subprocess.Popen(generate_tfrecord,shell=True,cwd='dataset/')
    generate_tfrecord_process.wait()
for datapath in datapaths:
    data_name = datapath.split('\\')[-1]
    output_dir = data_name.split('_')[-1][:-4]
    calciumgan_command = 'python main.py --input_dir dataset\\tfrecords\\sl2048_'+output_dir+' --output_dir runs\\'+output_dir+' --epochs 150 --batch_size 128 --model calciumgan --algorithm wgan-gp --noise_dim 32 --num_units 64 --kernel_size 24 --strides 2 --m 10 --layer_norm --mixed_precision --save_generated last'
    calcium_synthesis =  subprocess.Popen(calciumgan_command,shell=True)
    calcium_synthesis.wait()
    # if os.path.exists('dataset\\tfrecords\\sl2048'):
    #     shutil.rmtree('dataset\\tfrecords\\sl2048')
    # if os.path.exists('dataset\\tfrecords\\sl2048'):
    #     del_sl2048_command = 'rd/s/q dataset\\tfrecords\\sl2048'
    #     del_sl2048 = subprocess.Popen(del_sl2048_command,shell=True)
    #     del_sl2048.wait()