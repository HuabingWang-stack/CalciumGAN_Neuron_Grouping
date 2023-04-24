import os
import argparse
import h5py
import json
import pickle
import subprocess
from gan.utils import h5_helper
import numpy as np

def get_grouping_dict(hparams,kmeans =False):
    grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    if kmeans:
         grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    dict_path = os.path.join(grouping_info_path,'grouping_dict.json')
    if os.path.exists(dict_path):
        with open(dict_path, 'r') as jsonfile:
            neurons_grouping_dict =json.load(jsonfile)
        return neurons_grouping_dict

def split_whole_into_groups(hparams,grouping_dict,kmeans=False):
    if hparams.epochs <= 100:
        num_of_epochs = '0'+str(hparams.epochs-1)
    else:
        num_of_epochs = str(hparams.epochs-1)

    whole_data_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.serial)
    synthetic_data_path = os.path.join(whole_data_path,'generated','epoch'+num_of_epochs+'_signals.h5')
    validation_data_path = os.path.join(whole_data_path,'generated','validation.h5')
    synthetic_file = h5py.File(synthetic_data_path,'r')
    validation_file = h5py.File(validation_data_path,'r')
    for k,v in grouping_dict.items():
        if kmeans:
            group_path = whole_data_path+'_kg_'+str(k)
        else:
            group_path = whole_data_path+'_g_'+str(k)
        if os.path.exists(os.path.join(group_path,'metrics','plots_cascade')):
            continue
        synthetic_group_signals = np.take(synthetic_file['signals'],v,axis=2)
        synthetic_group_oasis = np.take(synthetic_file['spikes'],v,axis=2)
        synthetic_group_cascade = np.take(synthetic_file['cascade'],v,axis=2)
        validation_group_signals = np.take(validation_file['signals'],v,axis=2)
        validation_group_oasis = np.take(validation_file['spikes'],v,axis=2)
        validation_group_cascade = np.take(validation_file['cascade'],v,axis=2)
        
        os.makedirs(group_path, exist_ok=True)
        os.makedirs(os.path.join(group_path,'generated'), exist_ok=True)
        synthetic_group = h5py.File(os.path.join(group_path,'generated','epoch'+num_of_epochs+'_signals.h5'), 'w')
        synthetic_group.create_dataset('signals',data=synthetic_group_signals)
        synthetic_group.create_dataset('spikes',data=synthetic_group_oasis)
        synthetic_group.create_dataset('cascade',data=synthetic_group_cascade)
        synthetic_group.close()
        validation_group = h5py.File(os.path.join(group_path,'generated','validation.h5'), 'w')
        validation_group.create_dataset('signals',data=validation_group_signals)
        validation_group.create_dataset('spikes',data=validation_group_oasis)
        validation_group.create_dataset('cascade',data=validation_group_cascade)
        validation_group.close()
        with open(os.path.join(whole_data_path,'generated','info.pkl'), 'rb') as file:
            info = pickle.load(file)
        info[hparams.epochs-1]['filename'] = os.path.join(group_path,'generated',
        'epoch'+str(num_of_epochs)+'_signals.h5')
        with open(os.path.join(group_path,'generated','info.pkl'), 'wb') as handle:
            pickle.dump(info,handle)
        with open(os.path.join(whole_data_path,'hparams.json'), 'r') as jsonfile:
            hparams_dict = json.load(jsonfile)
        hparams_dict['input_dir'] = ''
        hparams_dict['output_dir'] = group_path
        hparams_dict['train_files'] = ''
        hparams_dict['validation_files'] = ''
        hparams_dict['signal_shape'] = [hparams_dict['signal_shape'][0],synthetic_group_signals.shape[2]]
        hparams_dict['spike_shape'] = [hparams_dict['spike_shape'][0],synthetic_group_oasis.shape[2]]
        hparams_dict['num_neurons'] = synthetic_group_signals.shape[2]
        hparams_dict['generated_dir'] = os.path.join(group_path,'generated')
        hparams_dict['validation_cache'] = os.path.join(group_path,'generated','validation.h5')
        json.dump(hparams_dict,open(os.path.join(group_path,'hparams.json'),'w'))

        compute_metrics_command = 'python compute_metrics_no_raster.py --output_dir '+group_path+' --num_neuron_plots '+str(synthetic_group_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric spikes'
        compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
        compute_metrics.wait()
        if hparams.spike_metric == 'cascade':
            compute_metrics_command = 'python compute_metrics_no_raster.py --output_dir '+group_path+' --num_neuron_plots '+str(synthetic_group_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+hparams.spike_metric
            compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
            compute_metrics.wait()
    synthetic_file.close()

def split_merged_into_groups(hparams,grouping_dict,kmeans=False):
    if hparams.epochs <= 100:
        num_of_epochs = '0'+str(hparams.epochs-1)
    else:
        num_of_epochs = str(hparams.epochs-1)
    if kmeans:
        data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    else:
        data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)

    merged_data_path = data_path+'_merged'
    synthetic_data_path = os.path.join(merged_data_path,'generated','epoch'+num_of_epochs+'_signals.h5')
    validation_data_path = os.path.join(merged_data_path,'generated','validation.h5')
    synthetic_file = h5py.File(synthetic_data_path,'r')
    validation_file = h5py.File(validation_data_path,'r')
    for k,v in grouping_dict.items():
        if kmeans:
            group_path = merged_data_path+'_kg_'+str(k)
        else:
            group_path = merged_data_path+'_g_'+str(k)
        if os.path.exists(os.path.join(group_path,'metrics','plots_cascade')):
            continue
        synthetic_group_signals = np.take(synthetic_file['signals'],v,axis=2)
        synthetic_group_oasis = np.take(synthetic_file['spikes'],v,axis=2)
        synthetic_group_cascade = np.take(synthetic_file['cascade'],v,axis=2)
        validation_group_signals = np.take(validation_file['signals'],v,axis=2)
        validation_group_oasis = np.take(validation_file['spikes'],v,axis=2)
        validation_group_cascade = np.take(validation_file['cascade'],v,axis=2)
        os.makedirs(group_path, exist_ok=True)
        os.makedirs(os.path.join(group_path,'generated'), exist_ok=True)
        synthetic_group = h5py.File(os.path.join(group_path,'generated','epoch'+num_of_epochs+'_signals.h5'), 'w')
        synthetic_group.create_dataset('signals',data=synthetic_group_signals)
        synthetic_group.create_dataset('spikes',data=synthetic_group_oasis)
        synthetic_group.create_dataset('cascade',data=synthetic_group_cascade)
        synthetic_group.close()
        validation_group = h5py.File(os.path.join(group_path,'generated','validation.h5'), 'w')
        validation_group.create_dataset('signals',data=validation_group_signals)
        validation_group.create_dataset('spikes',data=validation_group_oasis)
        validation_group.create_dataset('cascade',data=validation_group_cascade)
        validation_group.close()
        with open(os.path.join(merged_data_path,'generated','info.pkl'), 'rb') as file:
            info = pickle.load(file)
        info[hparams.epochs-1]['filename'] = os.path.join(group_path,'generated',
        'epoch'+str(num_of_epochs)+'_signals.h5')
        with open(os.path.join(group_path,'generated','info.pkl'), 'wb') as handle:
            pickle.dump(info,handle)
        with open(os.path.join(merged_data_path,'hparams.json'), 'r') as jsonfile:
            hparams_dict = json.load(jsonfile)
        hparams_dict['input_dir'] = ''
        hparams_dict['output_dir'] = group_path
        hparams_dict['train_files'] = ''
        hparams_dict['validation_files'] = ''
        hparams_dict['signal_shape'] = [hparams_dict['signal_shape'][0],synthetic_group_signals.shape[2]]
        hparams_dict['spike_shape'] = hparams_dict['spike_shape'][0],synthetic_group_oasis.shape[2]
        hparams_dict['num_neurons'] = synthetic_group_signals.shape[2]
        hparams_dict['generated_dir'] = os.path.join(group_path,'generated')
        hparams_dict['validation_cache'] = os.path.join(group_path,'generated','validation.h5')
        json.dump(hparams_dict,open(os.path.join(group_path,'hparams.json'),'w'))
        compute_metrics_command = 'python compute_metrics_no_raster.py --output_dir '+group_path+' --num_neuron_plots '+str(synthetic_group_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric spikes'
        compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
        compute_metrics.wait()
        if hparams.spike_metric == 'cascade':
            compute_metrics_command = 'python compute_metrics_no_raster.py --output_dir '+group_path+' --num_neuron_plots '+str(synthetic_group_signals.shape[2])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+hparams.spike_metric
            compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
            compute_metrics.wait()
    synthetic_file.close()
def main(hparams):
    grouping_dict = get_grouping_dict(hparams,kmeans=False)
    split_whole_into_groups(hparams,grouping_dict,kmeans=False)
    split_merged_into_groups(hparams,grouping_dict,kmeans=False)
    split_whole_into_groups(hparams,grouping_dict,kmeans=True)
    split_merged_into_groups(hparams,grouping_dict,kmeans=True)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs/80epochs')
    parser.add_argument('--spikename', default='cascade')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--serial', default='0')
    parser.add_argument('--epochs', default= 80,type=int)
    parser.add_argument('--num_processors', default=10, type=int)
    hparams = parser.parse_args()
    main(hparams)