import sys, os
import numpy


fname = sys.argv[-1]

if not os.path.isfile(fname):
    raise FileNotFoundError("Could not find file at ", {fname})

# Read the file into structured array:
metrics = numpy.genfromtxt(
    fname,
    delimiter=',',
    names=True,
)

timing_metrics = [ tm for tm in metrics.dtype.names if 'time' in tm]


means = []
stds  = []
for metric in timing_metrics:
    this_vals = metrics[metric][-25:]
    means.append(f"{numpy.mean(this_vals):.3f}")
    stds.append(f"{numpy.std(this_vals):.3f}")


print(",".join(timing_metrics))
print(",".join(means))
print(",".join(stds))
