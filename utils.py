import itertools
import matplotlib.pyplot as plt
import numpy as np
# Helper function to plot spiking activity
def plot_raster_rate(spikes_ex, spikes_in, rec_start, rec_stop, figsize=(9, 5)):
    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4, 0])
    ax2 = fig.add_subplot(gs[4, 0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])

    ax1.set_ylabel('Neuron ID')

    ax2.set_ylabel('Firing rate')
    ax2.set_xlabel('Time [ms]')

    for i in range(n_rec_in):
        ax1.plot(spikes_in[i],
                 i * np.ones(len(spikes_in[i])),
                 linestyle='',
                 marker='o',
                 color='r',
                 markersize=2)
    for i in range(n_rec_ex):
        ax1.plot(spikes_ex[i],
                 (i + n_rec_in) * np.ones(len(spikes_ex[i])),
                 linestyle='',
                 marker='o',
                 color='b',
                 markersize=2)

    ax2 = ax2.hist(spikes_ex_total,
                   range=(rec_start, rec_stop),
                   bins=int(rec_stop - rec_start))

    plt.savefig('raster.png')

    time_diff = (rec_stop - rec_start) / 1000.
    average_firing_rate = (len(spikes_total)
                           / time_diff
                           / (n_rec_ex + n_rec_in))
    print(f'Average firing rate: {average_firing_rate} Bq')


def average_isi(spike_trains, num_bins):
    '''
    Plot the average ISI distribution of multiple spike trains

    Paramters
    ---------

    spike_trains : list of np.ndarrays
        list of spike trains
    min_val : float
        minimal value of the range to calculate the distribution over
    max_val : float
        minimal value of the range to calculate the distribution over
    num_bins : int
        number of bins for the histogram of the distribution  
    '''
    isis = []
    for spike_train in spike_trains:
        isis.extend(spike_train[1:] - spike_train[:-1])
    plt.hist(isis, range=(np.min(isis), np.max(isis)), bins=num_bins)
    plt.savefig("isi.png")
    plt.close('all')
    ## If you want to, you can also calculate the CV here and print it direchtly from the function


def GetFiringFrequency(ti,N,simTime,dt=100.):#,startTime,endTime):,l,iteration,

    bins=np.arange(0,simTime,dt)
    hists,_ =np.histogram(ti,bins=bins)
    fireRate= np.zeros(len(bins))
    idx_count=1

    for n in hists:
#         print('n ', n,'\ndt ', dt,'\n N', N)
        fireRate[idx_count] =1000 * n / (simTime * N)
        idx_count+=1
    return fireRate,bins
    
    