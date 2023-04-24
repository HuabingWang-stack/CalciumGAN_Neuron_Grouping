import argparse
import subprocess
import os
def main(hparams):
    # os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpu
    for serial in range(hparams.start_serial,hparams.end_serial+1):

        run_as_a_whole_command = 'python run_as_a_whole.py --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        run_as_a_whole_process = subprocess.Popen(run_as_a_whole_command, shell=True)

        run_as_a_whole_process.wait()

        grouping_command = 'python GPFA_neuron_grouping.py --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        grouping_process = subprocess.Popen(grouping_command, shell=True)
        grouping_process.wait()

        grouping_command = 'python GPFA_neuron_grouping.py'+' --random_grouping --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        grouping_process = subprocess.Popen(grouping_command, shell=True)
        grouping_process.wait()

        grouping_command = 'python GPFA_neuron_grouping.py'+' --kmeans_cluster '+str(8)+' --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        grouping_process = subprocess.Popen(grouping_command, shell=True)
        grouping_process.wait()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs/80epochs')
    parser.add_argument('--spikename', default='cascade')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--epochs', default= 80,type=int)
    parser.add_argument('--start_serial', default=0, type=int)
    parser.add_argument('--end_serial', default=5, type=int)
    parser.add_argument('--num_processors', default=6, type=int)
    # parser.add_argument('--gpu', default='0')
    params = parser.parse_args() 
    main(params)