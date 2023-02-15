import argparse
import subprocess
import os
import pickle

def main(hparams):
    datapath = os.path.join('dataset','raw_data',hparams.dataname+'.pkl')
    with open(datapath, 'rb') as file:
        data = pickle.load(file)
    oasis_infer_command = 'python spike_train_inference.py --input_dir raw_data'
    oasis_infer_process = subprocess.Popen(oasis_infer_command,shell=True,cwd='dataset'+os.sep)
    oasis_infer_process.wait()
    generate_tfrecord = 'python generate_tfrecords.py --input raw_data/'+hparams.dataname+'.pkl --output_dir tfrecords/sl2048_'+hparams.dataname+'_'+hparams.serial+' --sequence_length 2048 --normalize'
    generate_tfrecord = subprocess.Popen(generate_tfrecord,shell=True,cwd='dataset'+os.sep)
    generate_tfrecord.wait()
    train_calciumgan = ' python main.py --input_dir dataset/tfrecords/sl2048_'+hparams.dataname+'_'+hparams.serial+' --output_dir '+os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.serial)+' --epochs '+str(hparams.epochs)+' --batch_size 128 --model calciumgan --algorithm wgan-gp --noise_dim 32 --num_units 64 --kernel_size 24 --strides 2 --m 10 --layer_norm --mixed_precision --save_generated last'
    train_calciumgan_process = subprocess.Popen(train_calciumgan,shell=True)
    train_calciumgan_process.wait()
    compute_metrics_command = 'python compute_metrics.py --output_dir '+os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.serial)+' --num_neuron_plots '+str(data['signals'].shape[0])+' --num_processors '+str(hparams.num_processors)+' --spike_metric '+ hparams.spike_metric
    compute_metrics_process = subprocess.Popen(compute_metrics_command,shell=True)
    compute_metrics_process.wait()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--spike_metric', default='oasis')
    parser.add_argument('--serial', default='0')
    parser.add_argument('--epochs', default= 150,type=int)
    parser.add_argument('--num_processors', default=6, type=int)
    params = parser.parse_args()
    main(params)