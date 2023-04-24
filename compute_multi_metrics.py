import os
import argparse
import h5py
import subprocess
from gan.utils import h5_helper

def main(hparams):
     
    if hparams.epochs <= 100:
        num_of_epochs = '0'+str(hparams.epochs-1)
    else:
        num_of_epochs = str(hparams.epochs-1)
    
    run_as_a_whole_0_datapath = os.path.join(hparams.output_dir, hparams.dataname+'_'+str(0))
    run_as_a_whole_0_validation = os.path.join(run_as_a_whole_0_datapath,'generated','validation.h5')
    run_as_a_whole_0_synthetic = os.path.join(run_as_a_whole_0_datapath,'generated','epoch'+num_of_epochs+'_signals.h5')
    if not os.path.exists(os.path.join(run_as_a_whole_0_datapath,'metrics','plots_'+hparams.spike_metric)):
        if os.path.exists(run_as_a_whole_0_validation) and os.path.exists(run_as_a_whole_0_synthetic):

            synthetic_signals = h5_helper.get(run_as_a_whole_0_synthetic,name='signals')
            if not h5_helper.contains(run_as_a_whole_0_validation ,name = hparams.spike_metric):
                print('spike metric '+hparams.spike_metric+' has not been inferred in '+run_as_a_whole_0_validation+' !')
            if not h5_helper.contains(run_as_a_whole_0_synthetic ,name = hparams.spike_metric):
                print('spike metric '+hparams.spike_metric+' has not been inferred in '+run_as_a_whole_0_synthetic+' !')
                compute_metrics_command = 'python compute_metrics.py --output_dir '+run_as_a_whole_0_datapath+' --num_neuron_plots '+str(synthetic_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+ hparams.spike_metric
                compute_metrics_process = subprocess.Popen(compute_metrics_command,shell=True)
                compute_metrics_process.wait()
            else:
                print('spike metric '+hparams.spike_metric+' is not computed in '+os.path.join(run_as_a_whole_0_datapath,'metrics','plots_'+hparams.spike_metric)+'!')
                compute_metrics_command = 'python compute_metrics.py --output_dir '+run_as_a_whole_0_datapath+' --num_neuron_plots '+str(synthetic_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+ hparams.spike_metric
                compute_metrics_process = subprocess.Popen(compute_metrics_command,shell=True)
                compute_metrics_process.wait()
        else:
            raise ValueError('synthetics signals has not been generated in '+run_as_a_whole_0_datapath+'!')
    
    for serial in range(hparams.start_serial,hparams.end_serial+1):
        run_as_a_whole_datapath = os.path.join(hparams.output_dir, hparams.dataname+'_'+str(serial))
        max_val_datapath = os.path.join(hparams.output_dir, hparams.dataname+'_cascade_GPFA_'+str(serial)+'_merged')
        random_datapath = os.path.join(hparams.output_dir, hparams.dataname+'_cascade_random_'+str(serial)+'_merged')
        kmeans_datapath = os.path.join(hparams.output_dir, hparams.dataname+'_cascade_GPFA_kmeans_'+str(serial)+'_merged')
                
        data_paths = [run_as_a_whole_datapath,max_val_datapath, random_datapath,kmeans_datapath]
        for datapath in data_paths:
            # if not os.path.exists(os.path.join(datapath,'metrics','plots_'+hparams.spike_metric)):
                synthetic_signal_path = os.path.join(datapath,'generated','epoch'+num_of_epochs+'_signals.h5')
                validation_path = os.path.join(datapath,'generated','validation.h5')
                if os.path.exists(synthetic_signal_path) and os.path.exists(validation_path):
                    if datapath == os.path.join(hparams.output_dir, hparams.dataname+'_'+str(0)):
                        pass
                    else:
                        validation = h5py.File(run_as_a_whole_0_validation,'r')
                        with h5py.File(os.path.join(datapath,'generated','validation.h5'),'a') as file:
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
                    if not h5_helper.contains(synthetic_signal_path,hparams.spike_metric):
                        print('spike metric '+hparams.spike_metric+' has not been inferred in '+synthetic_signal_path+'!')
                    synthetic_signals = h5_helper.get(synthetic_signal_path,name='signals')
                    compute_metrics_command = 'python compute_metrics.py --output_dir '+datapath+' --num_neuron_plots '+str(6)+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+ hparams.spike_metric
                    compute_metrics_process = subprocess.Popen(compute_metrics_command,shell=True)
                    compute_metrics_process.wait()
                else:
                    print('synthetics signals has not been generated in '+datapath+'!')
            # else:
            #     print('spike metric '+hparams.spike_metric+' has been computed in '+datapath+'!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--start_serial', default=0, type=int)
    parser.add_argument('--end_serial', default=0, type=int)
    parser.add_argument('--epochs', default= 150,type=int)
    parser.add_argument('--num_processors', default=10, type=int)
    hparams = parser.parse_args()
    main(hparams)