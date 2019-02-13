import numpy as np
import pylab as plt

def plot(name,m,d,mock,samples=None):
    plt.figure(figsize=(15,8))
    dist = m.domain[0].distances[0]
    npoints = m.domain[0].shape[0]
    xcoord = np.arange(npoints, dtype=np.float64)*dist
    plt.plot(xcoord,d.to_global_data(),'b.',
             label="data")
    plt.plot(xcoord,mock.to_global_data(),'g-',
             label="signal")
    plt.plot(xcoord,m.to_global_data(),'r-',
             label="reconstruction")
    if samples is not None:
        std=0.
        for s in samples:
            std = std + (s-m)**2
        std = std/len(samples)
        std = std.to_global_data()
        std = np.sqrt(std)
        md = m.to_global_data()
        plt.fill_between(xcoord,md-std,md+std,
                         alpha=0.3,color="k",
                         label=r"$1 \sigma$ uncertainty")
    plt.legend()
    x1, x2, y1, y2 = plt.axis()
    ymin = np.min(d.to_global_data()) - 0.1
    ymax = np.max(d.to_global_data()) + 0.1
    xmin = np.min(xcoord)
    xmax = np.max(xcoord)
    plt.axis((xmin,xmax,ymin,ymax))
    plt.savefig(name+".png",dpi=300)
    
def power_plot(name,s,m,samples=None):
    plt.figure(figsize=(15,8))
    ks = s.domain[0].k_lengths
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(ks,s.to_global_data(),'g-',label="true spectrum")
    plt.plot(ks,m.to_global_data(),'r-',label="rec. spectrum")
    if samples is not None:
       for i in range(len(samples)):
           if i==0:
               lgd = "samples"
           else:
               lgd = None
           plt.plot(ks,samples[i].to_global_data(),'k-',alpha=0.3,label=lgd)
    plt.legend()
    plt.savefig(name+".png",dpi=300)
