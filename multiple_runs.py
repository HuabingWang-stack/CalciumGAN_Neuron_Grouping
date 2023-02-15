import argparse
import subprocess
import os
def main(hparams):
    for serial in range(hparams.experiments):

        run_as_a_whole_command = 'python run_as_a_whole.py --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        run_as_a_whole_process = subprocess.Popen(run_as_a_whole_command)
        run_as_a_whole_process.wait()

        grouping_command = 'python GPFA_neuron_grouping.py --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        grouping_process = subprocess.Popen(grouping_command)
        grouping_process.wait()

        grouping_command = 'python GPFA_neuron_grouping.py'+' --kmeans_cluster '+str(8)+' --dataname '+hparams.dataname+' --output_dir '+hparams.output_dir +' --spike_metric '+hparams.spike_metric+' --epochs '+str(hparams.epochs)+' --num_processors '+str(hparams.num_processors)+ ' --serial '+str(serial)
        grouping_process = subprocess.Popen(grouping_command)
        grouping_process.wait()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', default='ST260_Day4')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--spikename', default='cascade')
    parser.add_argument('--spike_metric', default='spikes')
    parser.add_argument('--epochs', default= 150,type=int)
    parser.add_argument('--num_processors', default=12, type=int)
    parser.add_argument('--experiments', default=5, type=int)
    params = parser.parse_args()
    main(params)