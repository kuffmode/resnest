from network import *
import numpy as np
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
len_trial = 200.
g = 4.5
j = 0.1
n_trials = 200 #for each scenario so *2 per label

match_block = np.zeros((n_neurons, int(len_trial) + 1, n_trials*2))
nomatch_block = match_block.copy()

for trial in range(n_trials):

    dc_none = (100, 150, 0, 100, 150, 0)
    dc_both = (100, 150, 0.5, 100, 150, 0.5)
    dc_first = (100, 150, 0.5, 100, 150, 0)
    dc_second = (100, 150, 0, 100, 150, 0.5)

    ##================================================

    params_none = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': g,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc_none,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': j,
        'excitatory_inner_scale': 1.1
    }
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4})
    reservoir = BrunelNetwork(**params_none)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    match_block[:, :, trial] = prepare_trial(n_neurons,
                                             len_trial,
                                             spikes_ex,
                                             spikes_in)

    ##================================================
    params_both = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': g,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc_both,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': j,
        'excitatory_inner_scale': 1.1
    }
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4})
    reservoir = BrunelNetwork(**params_both)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    match_block[:, :, trial+n_trials] = prepare_trial(n_neurons,
                                                      len_trial,
                                                      spikes_ex,
                                                      spikes_in)

    ##================================================
    params_first = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': g,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc_first,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': j,
        'excitatory_inner_scale': 1.1
    }
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4})
    reservoir = BrunelNetwork(**params_first)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    nomatch_block[:,:,trial] = prepare_trial(n_neurons,
                                             len_trial,
                                             spikes_ex,
                                             spikes_in)

    ##================================================
    params_second = {
        'num_neurons': n_neurons,  # number of neurons in network
        'rho': 0.2,  # fraction of inhibitory neurons
        'eps': 0.1,  # probability to establish a connections
        'g': g,  # excitation-inhibition balanc
        'eta': 0.9,  # relative external rate
        'neuron_params': neuron_params,  # single neuron parameters
        'n_rec_ex': int(n_neurons * 0.8),  # excitatory neurons to be recorded from
        'n_rec_in': int(n_neurons * 0.2),  # inhibitory neurons to be recorded from
        'rec_start': 0.,  # start point for recording spike trains
        'rec_stop': len_trial,  # end points for recording spike trains
        'dc_params': dc_second,
        'w_mean': 0.005,
        'w_variance': 0.01,
        'w_threshold': 0.00001,
        'J': j,
        'excitatory_inner_scale': 1.1
    }
    nest.ResetKernel()
    nest.SetKernelStatus({'local_num_threads': 4})
    reservoir = BrunelNetwork(**params_second)
    reservoir.create()
    reservoir.simulate(t_sim=len_trial)
    spikes_ex, spikes_in = reservoir.get_data()

    nomatch_block[:,:,trial+n_trials] = prepare_trial(n_neurons,
                                                      len_trial,
                                                      spikes_ex,
                                                      spikes_in)

    ##================================================
match_responses = match_block[:,140:-1,:].copy()
nomatch_responses = nomatch_block[:,140:-1,:].copy()

with open('xor_match_responses_dc05.pkl', 'wb') as f:
    pickle.dump(match_responses, f, 1)

with open('xor_nomatch_responses_dc05.pkl', 'wb') as f:
    pickle.dump(nomatch_responses, f, 1)


with open('xor_match_block_dc05.pkl', 'wb') as f:
    pickle.dump(match_block, f, 1)

with open('xor_nomatch_block_dc05.pkl', 'wb') as f:
    pickle.dump(nomatch_block, f, 1)

