import os
import argparse
import GPFA_neuron_grouping
import pickle
import numpy as np
import subprocess
import glob
import h5py
import copy
import shutil

def split_data(hparams,data,grouping_dict):

    if hparams.kmeans_cluster:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)

    if os.path.exists(os.path.join(whole_data_path+'.pkl')):
        print('whole data has been merged!')
        return
    merged_data = {}
    for k,v in grouping_dict.items():
        
        splitted_data = {}
        splitted_data['signals'] = data['signals'][grouping_dict[k]]
        if hparams.spikename != 'oasis':
            splitted_data[hparams.spikename] = data[hparams.spikename][grouping_dict[k]]
        if len(data[hparams.spikename])==len(data['oasis']):
            splitted_data['oasis'] = data['oasis'][grouping_dict[k]]
        else:
            print(hparams.spikename+' does not have the same number of neurons with oasis!')
        if 'signals' not in merged_data.keys():
            merged_data['signals'] = splitted_data['signals']
        else:
            merged_data['signals'] = np.append(merged_data['signals'],splitted_data['signals'],axis = 0)
        if 'oasis' not in merged_data.keys():
            merged_data['oasis'] = splitted_data['oasis']
        else:
            merged_data['oasis'] = np.append(merged_data['oasis'],splitted_data['oasis'],axis = 0)
        if 'neuron_indicies' not in merged_data.keys():
            merged_data['neuron_indicies'] = np.array([])
        merged_data['neuron_indicies'] = np.append(merged_data['neuron_indicies'],v)



    with open(os.path.join(whole_data_path+'.pkl'),'wb') as handle:
        pickle.dump(merged_data, handle)
def run(hparams):
    if hparams.kmeans_cluster:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    
    datapath = os.path.join(whole_data_path+'.pkl')
    file_name_pkl = datapath.split(os.sep)[-1]
    file_name = file_name_pkl.split('.')[0]
    generate_tfrecord = 'python generate_tfrecords.py --input '+os.path.join('raw_datas',file_name_pkl)+' --output_dir '+os.path.join('tfrecords','sl2048_'+file_name)+' --sequence_length 2048 --normalize'
    generate_tfrecord_process = subprocess.Popen(generate_tfrecord,shell=True,cwd='dataset'+os.sep)
    generate_tfrecord_process.wait()
    if glob.glob(os.path.join(hparams.output_dir,file_name,'generated','*singals.h5')):
        print(file_name+' has been synthesized.')
    else:
        calciumgan_command = 'python main.py --input_dir '+os.path.join('dataset','tfrecords','sl2048_'+file_name)+' --output_dir '+os.path.join(hparams.output_dir,file_name)+' --epochs '+str(hparams.epochs)+' --batch_size 128 --model calciumgan2d --algorithm wgan-gp --noise_dim 32 --num_units 64 --kernel_size 24 --strides 2 --m 10 --layer_norm --mixed_precision --save_generated last'
        calcium_synthesis =  subprocess.Popen(calciumgan_command,shell=True)
        calcium_synthesis.wait()

def reconstruct_synthetic_data(hparams):
    if hparams.kmeans_cluster:
        synthetic_data_path = os.path.join(hparams.output_dir,
        'whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial) 
    elif hparams.random_grouping:
         synthetic_data_path = os.path.join(hparams.output_dir,
        'whole_'+hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        synthetic_data_path = os.path.join(hparams.output_dir,
        'whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    whole_synthetic_data_path = os.path.join(synthetic_data_path,'generated')

    if hparams.kmeans_cluster:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        whole_data_path = os.path.join('dataset','raw_datas','whole_'+hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)

    if not os.path.exists(whole_data_path+'.pkl') and os.path.exists(os.path.join(whole_synthetic_data_path,'validation.h5')):
        merged_data = pickle.load(open(whole_data_path+'.pkl','rb'))

        if hparams.epochs <= 100:
            num_of_epochs = '0'+str(hparams.epochs-1)
        else:
            num_of_epochs = str(hparams.epochs-1)
        
        synthetic_file =  h5py.File(os.path.join(whole_synthetic_data_path,
                            'epoch'+num_of_epochs+'_signals.h5'),'a')
        num_neurons = synthetic_file['signals'].shape[2]
        original_indices = merged_data['neuron_indicies']
        sorting_order = np.argsort(original_indices)
        if 'sorted' not in synthetic_file:
            synthetic_signals = synthetic_file['signals']
            sorted_synthetic_signal = np.take(synthetic_signals,sorting_order,axis = 2)
            synthetic_file.create_dataset('new_signals', data=sorted_synthetic_signal)
            del synthetic_file['signals']
            synthetic_file.move('new_signals','signals')
            synthetic_file['sorted'] = True
            if 'spikes' in synthetic_file.keys():
                synthetic_spikes =synthetic_file['spikes']
                sorted_synthetic_spikes = np.take(synthetic_spikes,sorting_order,axis = 2)
                synthetic_file.create_dataset('new_spikes', data=sorted_synthetic_spikes)
                del synthetic_file['spikes']
                synthetic_file.move('new_spikes','spikes')
        else:
            print('synthetic file already sorted')
        synthetic_file.close()

        # validation_file =  h5py.File(os.path.join(whole_synthetic_data_path,
        #                        'validation.h5'),'a')
        original_synthetic_data_path = os.path.join(hparams.output_dir,hparams.dataname+'_0','generated')
        # original_validation_file = h5py.File(os.path.join(original_synthetic_data_path,
        #                        'validation.h5'),'r')
        # if 'sorted' not in validation_file:
        #     validation_signals = validation_file['signals']
        #     sorted_validation_signal = np.take(validation_signals,sorting_order,axis = 2)
        #     validation_file.create_dataset('new_signals', data=sorted_validation_signal)
        #     del validation_file['signals']
        #     validation_file.move('new_signals','signals')
        #     validation_spikes =validation_file['spikes']
        #     sorted_validation_spikes = np.take(validation_spikes,sorting_order,axis = 2)
        #     validation_file.create_dataset('new_spikes', data=sorted_validation_spikes)
        #     del validation_file['spikes']
        #     validation_file.move('new_spikes','spikes')
        #     validation_file['sorted'] = True
        # original_validation_file.close()
        # num_neurons = validation_file['signals'].shape[2]
        # validation_file.close()
        validation_file_path = os.path.join(whole_synthetic_data_path, 'validation.h5')
        if os.path.isfile(validation_file_path):
            os.remove(validation_file_path)
        shutil.copyfile(os.path.join(original_synthetic_data_path,'validation.h5'),validation_file_path)

    compute_metrics_command = 'python compute_metrics.py --output_dir '+synthetic_data_path+' --num_neuron_plots '+str(102)+' --num_processors '+str(hparams.num_processors) + ' --spike_metric spikes'
    compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
    compute_metrics.wait()
    if hparams.spike_metric == 'cascade':
        compute_metrics_command = 'python compute_metrics.py --output_dir '+synthetic_data_path+' --num_neuron_plots '+str(102)+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+hparams.spike_metric
        compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
        compute_metrics.wait()

def main(hparams):
    data = GPFA_neuron_grouping.load_calcium_signals(hparams)
    data = GPFA_neuron_grouping.infer_spikes_by_cascade(hparams,data)
    matrix_of_spikeTrain = GPFA_neuron_grouping.get_each_trial_spiketrains(hparams,data)
    loading_matrix = GPFA_neuron_grouping.get_GPFA_loading_matrix(hparams,matrix_of_spikeTrain,None)
    if hparams.kmeans_cluster:
        grouping_dict = GPFA_neuron_grouping.get_grouping_dict_kmeans(hparams,loading_matrix,hparams.kmeans_cluster)
    elif hparams.random_grouping:
        grouping_dict = GPFA_neuron_grouping.get_random_grouping_dict(hparams)
    else:    
        grouping_dict = GPFA_neuron_grouping.get_grouping_dict(hparams,matrix_of_spikeTrain,loading_matrix)
    split_data(hparams,data,grouping_dict)
    run(hparams)
    reconstruct_synthetic_data(hparams)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs/80epochs')
    parser.add_argument('--spikename', default='cascade')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--kmeans_cluster', default=0, type=int)
    parser.add_argument('--random_grouping', action='store_true') # flaged to run random grouping as a reference group
    parser.add_argument('--epochs', default= 80,type=int)
    parser.add_argument('--serial', default='0') # append str to the run results's folder name to identify the run
    parser.add_argument('--num_processors', default=6, type=int)
    # parser.add_argument('--gpu', default='0')
    params = parser.parse_args() 
    main(params)