


def model_summary(writer, weights, step):
    return
    #TODO - tree flatten these
    for w, g in zip(weights, gradients):
        writer.add_histogram("weights/"   + w.name, w, global_step=step)

def wavefunction_summary(writer, latest_psi, step):
    writer.add_histogram("psi", latest_psi, global_step=step)


# @tf.function
def summary(writer, metrics, step):
    for key in metrics:
        writer.add_scalar(key, metrics[key], global_step=step)


def write_metrics(metric_file, metrics, step):
    '''
    Write the metrics into a csv file.
    '''
    if step == 0:
        # on the first step, dump the header in:
        metric_file.write("step,"+",".join(metrics.keys())+"\n")

    # Write the metrics in:
    values = [ f"{v:.5f}" for v in metrics.values()]
    metric_file.write(f"{step}," + ",".join(values)+"\n")
    metric_file.flush()
