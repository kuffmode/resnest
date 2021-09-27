from network import *
import numpy as np
from utils import plot_raster_rate, average_isi

import pickle
# Network parameters

def prepare_trial(n_neurons,
                  len_trial,
                  spikes_ex,
                  spikes_in):
    spikematrix = np.zeros((n_neurons, int(len_trial) + 1))
    all_spikes = np.hstack((spikes_ex, spikes_in))

    all_spikes_int = []
    for idx, spike_train in enumerate(all_spikes):
        all_spikes_int.append([int(spike) for spike in spike_train])

    for idx, spike_train in enumerate(all_spikes_int):
        for spike in spike_train:
            spikematrix[idx][spike] = 1
    return spikematrix


neuron_params = {"C_m": 1.0,
                 "tau_m": 20.,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": 20.}

n_neurons = 1_000
len_trial = 260.

n_trials = 500
dc_range = np.linspace(1, 4, n_trials)
match_block = np.zeros((n_neurons, int(len_trial) + 1, n_trials))
nomatch_block = match_block.copy()

for trial in range(n_trials):

    dc1start = np.random.randint(low=90, high=110)
    dc1stop = np.random.randint(low=140, high=160)
    dc2start = np.random.randint(low=140, high=160)
    dc2stop = np.random.randint(low=190, high=200)

    dc = (dc1start, dc1stop, dc_range[trial],
          dc2start, dc2stop, dc_range[trial])

    params = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': 4.5,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': 0.1,
        'excitatory_inner_scale': 1.1
    }

    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': -1})

    reservoir = BrunelNetwork(**params)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    match_block[:, :, trial] = prepare_trial(n_neurons,
                                             len_trial,
                                             spikes_ex,
                                             spikes_in)

    dc_nomatch = (dc1start, dc1stop, dc_range[trial],
                  dc2start, dc2stop, dc_range[trial] + 2)
    params = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': 4.5,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc_nomatch,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': 0.1,
        'excitatory_inner_scale': 1.1
    }
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4})

    reservoir = BrunelNetwork(**params)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    nomatch_block[:,:,trial] = prepare_trial(n_neurons,
                                             len_trial,
                                             spikes_ex,
                                             spikes_in)

match_responses = match_block[:,200:-1,:].copy()
nomatch_responses = nomatch_block[:,200:-1,:].copy()

with open('match_responses.pkl', 'wb') as f:
    pickle.dump(match_responses, f, 1)

with open('nomatch_responses.pkl', 'wb') as f:
    pickle.dump(nomatch_responses, f, 1)