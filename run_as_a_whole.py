import argparse
import subprocess
import os
import pickle
import h5py

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

    run_as_a_whole_validation_file = os.path.join(hparams.output_dir,hparams.dataname+'_0','generated','validation.h5')
    if os.path.exists(run_as_a_whole_validation_file):
        if hparams.serial == '0':
            pass
        else:
            validation = h5py.File(run_as_a_whole_validation_file, 'r')
            generated_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.serial)
            with h5py.File(os.path.join(generated_path,'generated','validation.h5'),'a') as file:
                if 'signals' in file.keys():
                    del file['signals']
                if 'signals' in validation.keys():
                    file.create_dataset('signals',shape = validation['signals'][:].shape, dtype= validation['signals'][:].dtype, data=validation['signals'][:] )
                if 'spikes' in file.keys():
                    del file['spikes']
                if 'spikes' in validation.keys():
                    file.create_dataset('spikes',validation['spikes'][:].shape, dtype= validation['spikes'][:].dtype, data=validation['spikes'][:])
                if 'cascade' in file.keys():
                    del file['cascade']
                if 'cascade' in validation.keys():
                    file.create_dataset('cascade',validation['cascade'][:].shape, dtype= validation['cascade'][:].dtype, data=validation['cascade'][:])
            validation.close()

    compute_metrics_command = 'python compute_metrics.py --output_dir '+os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.serial)+' --num_neuron_plots '+str(data['signals'].shape[0])+' --num_processors '+str(hparams.num_processors)+' --spike_metric '+ hparams.spike_metric
    compute_metrics_process = subprocess.Popen(compute_metrics_command,shell=True)
    compute_metrics_process.wait()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--serial', default='0')
    parser.add_argument('--epochs', default= 150,type=int)
    parser.add_argument('--num_processors', default=10, type=int)
    params = parser.parse_args()
    main(params)