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
    plt.axis((x1,x2,ymin,ymax))
    plt.savefig(name+".pdf",dpi=300)