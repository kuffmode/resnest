import sys
nest_path = '/Users/ICNS/nest/install/lib/python3.9/site-packages'
if nest_path not in sys.path:
    sys.path.append(nest_path)
import nest
import numpy as np

class BrunelNetwork:
    def __init__(self,
                 num_neurons,
                 rho,
                 eps,
                 g,
                 eta,
                 J,
                 neuron_params,
                 n_rec_ex, n_rec_in, rec_start, rec_stop,
                 dc_params, w_mean, w_variance, w_threshold,
                 excitatory_inner_scale):

        self.num_neurons = num_neurons
        self.num_ex = int((1 - rho) * num_neurons)  # number of excitatory neurons
        self.num_in = int(rho * num_neurons)  # number of inhibitory neurons
        self.c_ex = int(eps * self.num_ex)  # number of excitatory connections
        self.c_in = int(eps * self.num_in)  # number of inhibitory connections
        self.g = g
        self.n_rec_ex = n_rec_ex  # number of recorded excitatory neurons, both excitatory and inhibitory
        self.n_rec_in = n_rec_in  # number of recorded excitatory neurons, both excitatory and inhibitory
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.neuron_params = neuron_params  # neuron params
        self.ext_rate = (self.neuron_params['V_th']   # the external rate needs to be adapted to provide enough input
                         / (J * self.c_ex * self.neuron_params['tau_m'])
                         * eta * 1000. * self.c_ex)

        self.dc1_params = dc_params[:3] # (start, stop, amp)
        self.dc2_params = dc_params[3:]
        self.w_threshold = w_threshold
        self.weightmat = np.random.normal(w_mean, w_variance, (num_neurons, num_neurons))
        self.weightmat[self.weightmat <= self.w_threshold] = 0
        self.excitatory_inner_scale = excitatory_inner_scale
    def create(self):
        # Create the network

        # First create the neurons - the exscitatory and inhibitory populations together
        self.neurons_ = nest.Create("iaf_psc_alpha",
                                    self.num_neurons,
                                    params=self.neuron_params)



        self.excitatory_population_all = self.neurons_[:self.num_ex]
        self.inhibitory_population_all = self.neurons_[self.num_ex:]

        self.weightmat[self.num_ex:] = -self.g * self.weightmat[self.num_ex:]
        self.weightmat[:int(self.num_ex/2),:int(self.num_ex/2)] = \
            self.excitatory_inner_scale * self.weightmat[:int(self.num_ex/2),:int(self.num_ex/2)]
        self.weightmat[int(self.num_ex/2):self.num_ex,int(self.num_ex/2):self.num_ex] = \
            self.excitatory_inner_scale * self.weightmat[int(self.num_ex/2):self.num_ex,int(self.num_ex/2):self.num_ex]
        # from the first neuron to half of the excitatory population (35%) if the ratio is 70, 30
        self.excitatory_population1 = self.neurons_[:int(self.num_ex/2)] # receives dc1
        # the other half of the excitatory population
        self.excitatory_population2 = self.neurons_[int(self.num_ex/2):self.num_ex] # receives dc1


        # from the end of excitatory population to the half of inhibitory population (first 15%)
        self.inhibitory_population1 = self.neurons_[self.num_ex:int(self.num_ex+(self.num_in/2))] # receives dc2
        # the rest, second 15%
        self.inhibitory_population2 = self.neurons_[int(self.num_ex+(self.num_in/2)):] # receives dc2

        # Then create the external poisson spike generator
        self.spike_gen = nest.Create("poisson_generator",
                                     {"rate": self.ext_rate})

        self.dc1 = nest.Create("dc_generator")
        self.dc1.set(amplitude=self.dc1_params[2],
                     start=self.dc1_params[0],
                     stop=self.dc1_params[1])

        self.dc2 = nest.Create("dc_generator")
        self.dc2.set(amplitude=self.dc2_params[2],
                     start=self.dc2_params[0],
                     stop=self.dc2_params[1])

        # Then create spike detectors
        self.spike_rec_ex = nest.Create("spike_recorder",
                                        self.n_rec_ex,
                                        {"start": self.rec_start,
                                         "stop": self.rec_stop})

        self.spike_rec_in = nest.Create("spike_recorder",
                                        self.n_rec_in,
                                        {"start": self.rec_start,
                                         "stop": self.rec_stop})

        # Next we connect the excitatory and inhibitory neurons to each other, choose a delay of 1.5 ms
        nest.Connect(self.neurons_, self.neurons_,
                     {'rule':'all_to_all'},
                     {'delay': 1.5, 'weight': self.weightmat.T})

        # Then we connect the external drive to the neurons
        nest.Connect(self.spike_gen,
                   self.neurons_,
                   syn_spec={'weight': 0.1})



        nest.Connect(self.dc1,
                     self.excitatory_population1)
        nest.Connect(self.dc1,
                     self.inhibitory_population1)

        nest.Connect(self.dc2,
                     self.excitatory_population2)
        nest.Connect(self.dc2,
                     self.inhibitory_population2)
        # Then we connect the the neurons to the spike detectors
        # Note: You can use slicing for nest node collections as well

        nest.Connect(self.excitatory_population_all[:self.n_rec_ex], self.spike_rec_ex, 'one_to_one')
        nest.Connect(self.inhibitory_population_all[:self.n_rec_in:], self.spike_rec_in, 'one_to_one')

    def simulate(self, t_sim):
        # Simulate the network with specified
        nest.Simulate(t_sim)

    def get_data(self):
        '''
        Return spiking data from simulation

        Returns
        -------

        spikes_ex : list
            list of list of excitatory spike trains of n_rec_ex neurons

        spikes_in : list
            list of list of inhibitory spike trains of n_rec_in neurons
        '''
        # Define lists to store spike trains in
        spikes_ex = []
        spikes_in = []

        for neuron in nest.GetStatus(self.spike_rec_ex):
            spikes_ex.append(neuron['events']['times'])

        for neuron in nest.GetStatus(self.spike_rec_in):
            spikes_in.append(neuron['events']['times'])
        ## Your code here
        ## You can get the recorded quantities from the spike recorder with nest.GetStatus
        ## You may loop over the entries of the GetStatus return
        ## you might want to sort the spike times, they are not by default

        return spikes_ex, spikes_in
