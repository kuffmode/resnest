# resnest
This is a tiny project we did in EITN 2021 just to see how spiking neural networks work as a reservoir. The code is very messy and I gotta get back to it later so for now, it's just here as a reminder that I should really come back to it!
All you should care about at this moment are three files, network.py that has the network class in it and two XOR files. One simulates the network and the other does the decoding. Again, it's a mess, sorry!

<img src= "https://github.com/kuffmode/resnest/blob/main/Asset%201.jpg">
But briefly, the network is a population of excitatory/inhibitory neurons (N=1000) that receives two step functions simultaneously, a classifier (I tried both a logistic regression and a support vector machine) is trained on bins of size 10 ms, starting from the end of the stimulation to the end of the trial. The idea is to see how much information is still left in the "echo" after the stimulation is over and naturally it'll decade given the state of the network. I played with the excitatory/inhibitory balance "g" and as you can see, an excitatory dominated regime (g=1) has a longer memory than an almost balanced one (g=4.5). Interestingly, an inhibitory dominated regime also outperforms the balanced network probably because of the oscillations.

<img src= "https://github.com/kuffmode/resnest/blob/main/results.png">
