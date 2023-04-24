import os
import argparse
import numpy as np
import pickle
from Cascade.cascade2p import cascade
from Cascade.cascade2p.utils_discrete_spikes import infer_discrete_spikes
import multiprocessing
import neo
import quantities as pq
from elephant.gpfa import GPFA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import glob
import subprocess
import h5py
import warnings

def get_cumulative_shared_covariance(loading_matrix):
    """
    Calculate cumulative shared variance of loading matrix

    Adapted from viziphant.plot_cumulative_shared_covariance
    https://viziphant.readthedocs.io/en/latest/toctree/gpfa/
    Copyright (c) 2017-2022, Institute of Neuroscience and Medicine (INM-6), 
    Forschungszentrum Jülich All rights reserved.
    """
    eigenvalues = np.linalg.eigvals(np.dot(loading_matrix.transpose(),
                                        loading_matrix))
    total_variance = np.sum(eigenvalues)
    sorted_eigenvalues = np.sort(np.abs(eigenvalues))[-1::-1]
    cumulative_variance = np.cumsum(sorted_eigenvalues / total_variance)
    return cumulative_variance

def plot_transform_matrix(loading_matrix, cmap='RdYlGn'):
    """
    This function visualizes the loading matrix as a heatmap.

    Adapted from viziphant.plot_cumulative_shared_covariance
    https://viziphant.readthedocs.io/en/latest/toctree/gpfa/
    Copyright (c) 2017-2022, Institute of Neuroscience and Medicine (INM-6), 
    Forschungszentrum Jülich All rights reserved.

    Parameters
    ----------
    loading_matrix : np.ndarray
        The loading matrix defines the mapping between neural space and
        latent state space. It is obtained by fitting a GPFA model and
        stored in ``GPFA.params_estimated['C']`` or if orthonormalized
        ``GPFA.params_estimated['Corth']``.
    cmap : str, optional
        Matplotlib imshow colormap.
        Default: 'RdYlGn'

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes

    """

    fig, axes = plt.subplots()

    vmax = np.max(np.abs(loading_matrix))
    vmin = -vmax

    heatmap = axes.imshow(loading_matrix,vmin=vmin, vmax=vmax,aspect='auto',
                          interpolation='none', cmap=cmap)

    axes.set_title('Loading Matrix')
    axes.set_ylabel('Neuron ID')
    axes.set_xlabel('Latent Variable')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar = plt.colorbar(heatmap, cax=cax)
    colorbar.set_label('Latent Variable Weight')

    return fig, axes

def load_calcium_signals(hparams):
    datapath = os.path.join('dataset','raw_data',hparams.dataname+'.pkl')
    with open(datapath, 'rb') as file:
        data = pickle.load(file)
    # if platform.system() == 'Linux':
    #     splitter = '/'
    # if platform.system() == 'Windows':
    #     splitter = '\\'
    # data_name = datapath.split('.')[0]
    return data

def infer_spikes_by_cascade(hparams,data):

    if 'cascade' in data.keys():
        print("data['signals'] already has cascade inferred spike trains!")
        return data

    model_name = "Global_EXC_25Hz_smoothing50ms_causalkernel"
    if not os.path.exists(os.path.join("Cascade","Pretrained_models",
    model_name)):
        cascade.download_model( model_name = model_name ,model_folder = "Cascade/Pretrained_models",verbose = 1)
    total_array_size = data['signals'].itemsize*data['signals'].size*64/1e9
    if total_array_size < 6:
        spike_prob = cascade.predict(model_name, data['signals'],model_folder="Cascade/Pretrained_models", verbosity=1)
    else:
        from tqdm import trange
        print("Split analysis into chunks in order to fit into memory.")

        # pre-allocate array for results
        spike_prob = np.zeros((data['signals'].shape))
        # nb of neurons and nb of chuncks
        nb_neurons = data['signals'].shape[0]
        nb_chunks = np.int(np.ceil(total_array_size/5))

        chunks = np.array_split(range(nb_neurons), nb_chunks)
        # infer spike rates independently for each chunk
        for part_array in trange(nb_chunks):
            spike_prob[chunks[part_array],:] = cascade.predict( 
                model_name, data['signals'][chunks[part_array],:] )
    
    pool = multiprocessing.Pool(hparams.num_processors)
    results = pool.starmap(infer_discrete_spikes,
    [(np.expand_dims(spike_prob[n],axis=0),model_name,"Cascade/Pretrained_models") for n in range(len(spike_prob))])
    cascade_spike = np.zeros((data['signals'].shape[0],data['signals'].shape[1]))
    for n in range(len(results)):
        spike_time_of_n = results[n][1][0]
        cascade_spike[n][spike_time_of_n] = 1
        data['cascade'] = cascade_spike
    with open(os.path.join('dataset','raw_data',hparams.dataname+'.pkl'), 'wb') as handle:
        pickle.dump(data,handle)
    return data

def get_each_trial_spiketrains(hparams,data):
    spikedata = data[hparams.spikename]
    trial = np.array(data['trial'])
    matrix_of_spikeTrain = []
    num_of_trials = np.max(trial)+1
    num_of_neurons = len(spikedata)
    for t in range(0,num_of_trials):
        list_of_spikeTrain = []
        spikeTrains_time = spikedata[:,trial==t]
        for i in range(0,num_of_neurons):
            spikeTrain_time = np.array(spikeTrains_time[i])
            trail_hz = len(spikeTrain_time)
            spike_indice = np.where(spikeTrain_time==1)[0]
            spike_indice_ms = spike_indice *(1000/24)
            list_of_spikeTrain.append(neo.SpikeTrain(
                spike_indice_ms*pq.ms,axis=1*pq.ms,
                t_start = 0*pq.ms,t_stop=trail_hz *(1000/24)*pq.ms))
        matrix_of_spikeTrain.append(list_of_spikeTrain)
    return matrix_of_spikeTrain


def get_GPFA_loading_matrix(hparams, matrix_of_spikeTrain,reduced_dim):
    if hparams.kmeans_cluster:
        fig_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    else:
        fig_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    dict_path = os.path.join(fig_path,'grouping_dict.json')

    if os.path.exists(dict_path):
        return None

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    
    warnings.filterwarnings(action='ignore',message="Binning discarded", category=UserWarning)
    # if not os.path.exists(os.path.join(fig_path,'cummulative_shared_covariance.png')):
    if reduced_dim == None:
        gpfa_10dim = GPFA(bin_size=100*pq.ms, x_dim=10)
        gpfa_10dim.fit(matrix_of_spikeTrain)
        loading_matrix = gpfa_10dim.params_estimated['C']
        # plot cummulative covariance when dim-reduce to 10 dimension
        cumulative_variance = get_cumulative_shared_covariance(loading_matrix)
        fig, axes = plt.subplots()
        axes.plot(cumulative_variance, 'o-')
        axes.grid()
        axes.set_title('Eigenspectrum of estimated shared covariance matrix')
        axes.set_xlabel('Latent Dimensionality')
        axes.set_ylabel('Cumulative % of shared variance')
        fig.savefig(os.path.join(fig_path,'cummulative_shared_covariance.png'))

        sufficient_dim = np.argmax(cumulative_variance>0.95)+1

    if reduced_dim != None:
        print('reduced to '+str(reduced_dim)+' dimensions!')
        sufficient_dim = reduced_dim
    # save loading matrix 
    gpfa_suff_dim = GPFA(bin_size=100*pq.ms, x_dim=sufficient_dim)
    gpfa_suff_dim.fit(matrix_of_spikeTrain)
    loading_matrix = gpfa_suff_dim.params_estimated['C']
    return loading_matrix

def get_grouping_dict(hparams,matrix_of_spikeTrain,loading_matrix):

    grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    dict_path = os.path.join(grouping_info_path,'grouping_dict.json')

    if os.path.exists(dict_path):
        with open(dict_path, 'r') as jsonfile:
            neurons_grouping_dict =json.load(jsonfile)
        return neurons_grouping_dict

    max_val_per_dim_loading_matrix = []
    neurons_grouping_dict = {}

    for neuron_idx in range(len(loading_matrix)):
        neuron_loadings = loading_matrix[neuron_idx]
        neurons_loading_value_at_class = np.zeros(neuron_loadings.shape[0])
        largest_loading_latent_dim = int(np.argmax(np.abs(neuron_loadings)))
        neurons_loading_value_at_class[largest_loading_latent_dim] = neuron_loadings[
            largest_loading_latent_dim]
        max_val_per_dim_loading_matrix.append(neurons_loading_value_at_class)
        if largest_loading_latent_dim in neurons_grouping_dict.keys():
            neurons_grouping_dict[largest_loading_latent_dim].append(neuron_idx)
        else:
            neurons_grouping_dict[largest_loading_latent_dim] = [neuron_idx]

    # if number of neurons in one of latent dimension smaller than 4 (<4 will make 
    # synthetic signals hard to initialize), reduce one dimension and compute loading matrix again
    if any(len(neurons)<=4 for neurons in neurons_grouping_dict.values()):
        loading_matrix = get_GPFA_loading_matrix(
            hparams,matrix_of_spikeTrain,len(neurons_grouping_dict.keys())-1)
        print('Invalid Grouping Dict:')
        print(neurons_grouping_dict)
        return get_grouping_dict(hparams,matrix_of_spikeTrain,loading_matrix)
    
    fig, axes = plot_transform_matrix(loading_matrix)
    fig.savefig(os.path.join(grouping_info_path,'C_{}_dim.png'.format(len(neurons_grouping_dict.keys()))))
    fig, axes = plot_transform_matrix(max_val_per_dim_loading_matrix)
    fig.savefig(os.path.join(grouping_info_path,'C_{}_dim_only_max.png'.format(len(neurons_grouping_dict.keys()))))
    json.dump(neurons_grouping_dict, open(dict_path,'w'))
    return neurons_grouping_dict

def get_random_grouping_dict(hparams):

    grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    dict_path = os.path.join(grouping_info_path,'grouping_dict.json')
    max_val_grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    max_val_dict_path = os.path.join(max_val_grouping_info_path,'grouping_dict.json')

    if not os.path.exists(grouping_info_path):
        os.makedirs(grouping_info_path)

    if not os.path.exists(max_val_dict_path):
        print('grouping dict created by max loading value does not exist!')
        return None
    with open(max_val_dict_path, 'r') as jsonfile:
        grouping_dict_max_val =json.load(jsonfile)
    max_neuron = max(num for arr in grouping_dict_max_val.values() for num in arr)
    neurons_list = list(range(max_neuron+1))
    grouping_dict = {}
    sorted_grouping_dict_max_val = dict(sorted(grouping_dict_max_val.items()))
    for k,v in sorted_grouping_dict_max_val.items():
        grouping_dict[k] = neurons_list[:len(v)]
        neurons_list = neurons_list[len(v):]
    json.dump(grouping_dict, open(dict_path,'w'))
    return grouping_dict

def get_grouping_dict_kmeans(hparams,loading_matrix,initial_groups):

    grouping_info_path = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    dict_path = os.path.join(grouping_info_path,'grouping_dict.json')

    if os.path.exists(dict_path):
        with open(dict_path, 'r') as jsonfile:
            neurons_grouping_dict =json.load(jsonfile)
        return neurons_grouping_dict
    
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    neurons_grouping_dict = {}
    while any(len(neurons)<=4 for neurons in neurons_grouping_dict.values()) or neurons_grouping_dict == {}:
        neurons_grouping_dict = {}
        print('reduced to '+str(initial_groups)+' groups!')
        Kmean = KMeans(n_clusters=initial_groups)
        Kmean.fit(loading_matrix)
        for i in range(len(Kmean.labels_)):
            if Kmean.labels_[i] in neurons_grouping_dict.keys():
                neurons_grouping_dict[int(Kmean.labels_[i])].append(i)
            else:
                neurons_grouping_dict[int(Kmean.labels_[i])] = [i]
        initial_groups-=1
    
    from sklearn.decomposition import PCA
    pca = PCA()
    Ct = pca.fit_transform(loading_matrix)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    existing_label = set()
    clist = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    for i in range(len(Ct)):
        if Kmean.labels_[i] not in existing_label:
            existing_label.add(Kmean.labels_[i])
            ax.scatter(Ct[i,0],Ct[i,1], Ct[i,2],c=clist[Kmean.labels_][i],label='class'+str(Kmean.labels_[i]))
        else:
            ax.scatter(Ct[i,0],Ct[i,1], Ct[i,2],c=clist[Kmean.labels_][i])
    ax.set_xlabel('PC1 explained var {:.0f}%'.format(pca.explained_variance_ratio_[0]*100),fontsize=15)
    ax.set_ylabel('PC2 explained var {:.0f}%'.format(pca.explained_variance_ratio_[1]*100),fontsize=15)
    ax.set_zlabel('PC3 explained var {:.0f}%'.format(pca.explained_variance_ratio_[2]*100),fontsize=15)
    ax.set_title('PCA of loading values per-neuron',fontsize=20)
    ax.legend(fontsize=15)
    fig.savefig(os.path.join(grouping_info_path,'pca_of_loading_matrix.png'))
    json.dump(neurons_grouping_dict, open(dict_path,'w'))
    return neurons_grouping_dict

def split_data(hparams,data,grouping_dict):
    if hparams.kmeans_cluster:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)

    if os.path.exists(os.path.join(splitted_data_path+'_'+str(max(grouping_dict.keys()))+'.pkl')):
        print('data has been splitted!')
        return
    for k,v in grouping_dict.items():
        
        splitted_data = {}
        splitted_data['signals'] = data['signals'][grouping_dict[k]]
        if hparams.spikename != 'oasis':
            splitted_data[hparams.spikename] = data[hparams.spikename][grouping_dict[k]]
        if len(data[hparams.spikename])==len(data['oasis']):
            splitted_data['oasis'] = data['oasis'][grouping_dict[k]]
        else:
            print(hparams.spikename+' does not have the same number of neurons with oasis!')
        
        if not os.path.exists(os.path.join('dataset','raw_datas')):
            os.makedirs(os.path.join('dataset','raw_datas'))

        with open(os.path.join(splitted_data_path+'_'+str(k)+'.pkl'),'wb') as handle:
            pickle.dump(splitted_data, handle)

def generate_tfrecords_and_run(hparams):

    if hparams.kmeans_cluster:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        splitted_data_path = os.path.join('dataset','raw_datas',hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)

    datapaths = glob.glob(os.path.join(splitted_data_path+'*.pkl'))
    # if the last synthetic data file exists, pass this function
        
    for datapath in datapaths:
        file_name_pkl = datapath.split(os.sep)[-1]
        file_name = file_name_pkl.split('.')[0]
        # group_number = file_name.split('_')[-1][:-4]
        generate_tfrecord = 'python generate_tfrecords.py --input '+os.path.join('raw_datas',file_name_pkl)+' --output_dir '+os.path.join('tfrecords','sl2048_'+file_name)+' --sequence_length 2048 --normalize'
        generate_tfrecord_process = subprocess.Popen(generate_tfrecord,shell=True,cwd='dataset'+os.sep)
        generate_tfrecord_process.wait()
    for datapath in datapaths:
        file_name_pkl = datapath.split(os.sep)[-1]
        file_name = file_name_pkl.split('.')[0]
        if glob.glob(os.path.join(hparams.output_dir,file_name,'generated','*singals.h5')):
            print(file_name+' has been synthesized.')
            continue
        # group_number = data_name.split('_')[-1][:-4]
        calciumgan_command = 'python main.py --input_dir '+os.path.join('dataset','tfrecords','sl2048_'+file_name)+' --output_dir '+os.path.join(hparams.output_dir,file_name)+' --epochs '+str(hparams.epochs)+' --batch_size 128 --model calciumgan --algorithm wgan-gp --noise_dim 32 --num_units 64 --kernel_size 24 --strides 2 --m 10 --layer_norm --mixed_precision --save_generated last'
        calcium_synthesis =  subprocess.Popen(calciumgan_command,shell=True)
        calcium_synthesis.wait()

def merge_synthetic_data(hparams,grouping_dict):
    
    if hparams.kmeans_cluster:
        synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
         synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    merged_synthetic_data_path = os.path.join(synthetic_data_path+'_merged','generated')

    if not os.path.exists(merged_synthetic_data_path):
        os.makedirs(merged_synthetic_data_path)
    
    if hparams.epochs <= 100:
        num_of_epochs = '0'+str(hparams.epochs-1)
    else:
        num_of_epochs = str(hparams.epochs-1)

    if os.path.exists(os.path.join(merged_synthetic_data_path,
    'epoch'+num_of_epochs+'_signals.h5')) and os.path.exists(os.path.join(
        merged_synthetic_data_path,'validation.h5')):
        return 

    max_neuron = 0
    for k,v in grouping_dict.items():
        if any(np.array(v)>max_neuron):
            max_neuron = max(v)
    first_looped_neuron_flag = True
    for neuron_idx in range(max_neuron+1):
        for k,v in grouping_dict.items():
            if neuron_idx in v:
                latent_dim = k
                idx_in_loaded_file = v.index(neuron_idx)

        synthetic_file = h5py.File(os.path.join(synthetic_data_path+'_'+str(latent_dim),
        'generated','epoch'+num_of_epochs+'_signals.h5'),'r')
        validation_file = h5py.File(os.path.join(synthetic_data_path+'_'+str(latent_dim),
        'generated','validation.h5'),'r')
        if first_looped_neuron_flag:
            merged_signals = np.array(synthetic_file['signals'][:,:,idx_in_loaded_file][...,None])
            merged_validation_signals = np.array(validation_file['signals'][:,:,idx_in_loaded_file][...,None])
            merged_validation_spikes = np.array(validation_file['spikes'][:,:,idx_in_loaded_file][...,None])
            first_looped_neuron_flag = False
        else:
            merged_signals = np.append(merged_signals,
                synthetic_file['signals'][:,:,idx_in_loaded_file][...,None],axis=2)
            merged_validation_signals = np.append(merged_validation_signals,
                validation_file['signals'][:,:,idx_in_loaded_file][...,None],axis=2)
            merged_validation_spikes = np.append(merged_validation_spikes,
                validation_file['spikes'][:,:,idx_in_loaded_file][...,None],axis=2)
        synthetic_file.close()
        validation_file.close()

    hf = h5py.File(os.path.join(merged_synthetic_data_path,'epoch'+num_of_epochs+'_signals.h5'), 'w')
    hf.create_dataset('signals',data=merged_signals)
    hf.close()
    vf = h5py.File(os.path.join(merged_synthetic_data_path,'validation.h5') , 'w')
    vf.create_dataset('signals',data=merged_validation_signals)
    vf.create_dataset('spikes',data=merged_validation_spikes)
    vf.close()

def create_info_and_hparams(hparams,data):

    if hparams.kmeans_cluster:
        synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_kmeans_'+hparams.serial)
    elif hparams.random_grouping:
        synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_random_'+hparams.serial)
    else:
        synthetic_data_path = os.path.join(hparams.output_dir,
        hparams.dataname+'_'+hparams.spikename+'_GPFA_'+hparams.serial)
    merged_synthetic_data_path = synthetic_data_path+'_merged'

    if hparams.epochs <= 100:
        num_of_epochs = '0'+str(hparams.epochs-1)
    else:
        num_of_epochs = str(hparams.epochs-1)

    path_of_group_0 = os.path.join(synthetic_data_path+'_'+str(0))
    # path_of_merged_signals = os.path.join(hparams.output_dir,hparams.dataname+'_'+hparams.spikename+'_'+'merged')
    with open(os.path.join(path_of_group_0,'generated','info.pkl'), 'rb') as file:
        info = pickle.load(file)
    info[hparams.epochs-1]['filename'] = os.path.join(merged_synthetic_data_path,'generated',
    'epoch'+str(num_of_epochs)+'_signals.h5')
    with open(os.path.join(merged_synthetic_data_path,'generated','info.pkl'), 'wb') as handle:
        pickle.dump(info,handle)
    with open(os.path.join(path_of_group_0,'hparams.json'), 'r') as jsonfile:
        hparams_dict = json.load(jsonfile)
    hparams_dict['input_dir'] = ''
    hparams_dict['output_dir'] = merged_synthetic_data_path
    hparams_dict['train_files'] = ''
    hparams_dict['validation_files'] = ''
    hparams_dict['signal_shape'] = [hparams_dict['signal_shape'][0],data['signals'].shape[0]]
    hparams_dict['spike_shape'] = [hparams_dict['spike_shape'][0],data['signals'].shape[0]]
    hparams_dict['num_neurons'] = data['signals'].shape[0]
    hparams_dict['generated_dir'] = os.path.join(merged_synthetic_data_path,'generated')
    hparams_dict['validation_cache'] = os.path.join(merged_synthetic_data_path,'generated','validation.h5')
    json.dump(hparams_dict,open(os.path.join(merged_synthetic_data_path,'hparams.json'),'w'))

    # if ST260_Day4_0 exists for example, we directly use signals and inferred cascade spikes from 
    # ST260_Day4_0/generated/validation.h5
    run_as_a_whole_validation_file = os.path.join(hparams.output_dir,hparams.dataname+'_0','generated','validation.h5')
    if os.path.exists(run_as_a_whole_validation_file):
        validation = h5py.File(run_as_a_whole_validation_file, 'r')
        with h5py.File(os.path.join(merged_synthetic_data_path,'generated','validation.h5'),'a') as file:
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
    compute_metrics_command = 'python compute_metrics.py --output_dir '+merged_synthetic_data_path+' --num_neuron_plots '+str(data['signals'].shape[0])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric spikes'
    compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
    compute_metrics.wait()
    if hparams.spike_metric == 'cascade':
        compute_metrics_command = 'python compute_metrics.py --output_dir '+merged_synthetic_data_path+' --num_neuron_plots '+str(data['signals'].shape[0])+' --num_processors '+str(hparams.num_processors) + ' --spike_metric '+hparams.spike_metric
        compute_metrics =  subprocess.Popen(compute_metrics_command,shell=True)
        compute_metrics.wait()
        
def main(hparams):
    data = load_calcium_signals(hparams)
    data = infer_spikes_by_cascade(hparams,data)
    matrix_of_spikeTrain = get_each_trial_spiketrains(hparams,data)
    loading_matrix = get_GPFA_loading_matrix(hparams,matrix_of_spikeTrain,None)
    if hparams.kmeans_cluster:
        grouping_dict = get_grouping_dict_kmeans(hparams,loading_matrix,hparams.kmeans_cluster)
    elif hparams.random_grouping:
        grouping_dict = get_random_grouping_dict(hparams)
    else:    
        grouping_dict = get_grouping_dict(hparams,matrix_of_spikeTrain,loading_matrix)
    split_data(hparams,data,grouping_dict)
    generate_tfrecords_and_run(hparams)
    merge_synthetic_data(hparams,grouping_dict)
    create_info_and_hparams(hparams,data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs/80epochs')
    parser.add_argument('--kmeans_cluster', default=0, type=int)
    parser.add_argument('--random_grouping', action='store_true') # flaged to run random grouping as a reference group
    parser.add_argument('--spikename', default='cascade')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--serial', default='0') # append str to the run results's folder name to identify the run
    parser.add_argument('--epochs', default= 80 ,type=int)
    parser.add_argument('--num_processors', default=10, type=int)
    params = parser.parse_args()
    main(params)