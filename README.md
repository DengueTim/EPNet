# Experimenting with Visual Flow Estimation CNNs..

Given most current approaches to visual flow estimation using CNNs try to
reduce the computational cost and memory requirement of using a
cost volume(or some scaled/hierarchical representation). Is
it worth reassessing an approach more like FlowNetSimple, or the use of a
distributed representation to approximate the cost volume? 
Looking at the results of the FlowNet paper the Simple network
only marginally under-performs the Correlated network while having almost half the 
runtime. Part of the power of ANNs is in their distributed representations after all...


### Rough Tests...

Comparison using Scene Flow datasets scaled down by ~15 times.  After 10 passes.

| Network       | Params        | EPE  |
| ------------- |:-------------:| -----:|
| FlowNetS      | 0.6m          | 0.238 |
| FlowNetC      | 10m           | 0.259 |
| PWCNet        | 4.5m          |   0.3 |
| MySimpleNet   |  5m           |  0.35 |
| MyCorrNet(EcD) |  0.45m     | 0.374 |

### Observations/Ideas

- The ability of the net to average the flow estimation out between neighboring pixels
seems a major factor for better results.  Is this at the cost of estimating fine detail?
  Training was probably too short here.  Think average (+ residual)<sup>n</sup>
    
- FlowNetS concatenates the input image colour channels and feeds that into the network.
  Is there value in extracting features from the input images first and then feeding
  those into a network to estimate flow?  FlowNetS vs MySimpleNet suggests not, but
  MySimpleNet doesn't have the ability to uses as many neighboring pixels. 