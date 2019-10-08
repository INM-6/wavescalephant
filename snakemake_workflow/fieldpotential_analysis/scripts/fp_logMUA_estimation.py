import numpy as np
import elephant as el
import neo
import argparse
import os

def detrending(asig, order):
    # ToDo: Improve algorithm and include into elephant
    # ToDo: return neo.Analogsignal
    X = asig.as_array()
    window_size = len(asig)
    if order > 0:
        X = X - np.mean(X, axis=0)
    if order > 1:
        factor = [1, 1/2., 1/6.]
        for i in np.arange(order-1)+1:
            detrend = np.linspace(-window_size/2., window_size/2., window_size)**i \
                      * np.mean(np.diff(X, n=i, axis=0)) * factor[i-1]
            X = X - detrend
    return X


def build_logMUA_segment(segment, freq_band, detrending_order, psd_overlap):
    asig = segment.analogsignals[0]

    fs = asig.sampling_rate.rescale('1/s')
    FFTWindowSize = int(round(fs.magnitude / freq_band[0]))
    sample_num = int(np.floor(len(asig)/FFTWindowSize))
    MUA_sampling_rate = sample_num / (asig.t_stop - asig.t_start)

    logMUA_segment = neo.core.Segment(name='Segment logMUA')

    for asig in segment.analogsignals:
        logMUA_asig = logMUA_estimation(asig, fs, sample_num, FFTWindowSize,
                                        freq_band, MUA_sampling_rate,
                                        detrending_order, psd_overlap)
        logMUA_segment.analogsignals.append(logMUA_asig)

    return logMUA_segment


def logMUA_estimation(analogsignal, fs, sample_num, FFTWindowSize, freq_band,
                      MUA_sampling_rate, detrending_order, psd_overlap):
    MUA = np.zeros(sample_num)
    # calculating mean spectral power in each window
    for i in range(sample_num):
        local_asig = analogsignal[i*FFTWindowSize:(i+1)*FFTWindowSize]
        local_asig = detrending(local_asig, detrending_order)
        (f, p) = el.spectral.welch_psd(np.squeeze(local_asig),
                                       freq_res=freq_band[0],
                                       overlap=psd_overlap,
                                       window='hanning',
                                       nfft=None,
                                       fs=fs,
                                       detrend=False,
                                       return_onesided=True,
                                       scaling='density',
                                       axis=-1)
        low_idx = np.where(freq_band[0] <= f)[0][0]
        high_idx = np.where(freq_band[1] <= f)[0][0]
        MUA[i] = np.mean(p[low_idx:high_idx])

    logMUA_asig = neo.core.AnalogSignal(np.log(MUA),
                                        units='dimensionless',
                                        t_start=analogsignal.t_start,
                                        t_stop=analogsignal.t_stop,
                                        sampling_rate=MUA_sampling_rate,
                                        FFTWindowSize=FFTWindowSize,
                                        freq_band=freq_band,
                                        detrending_order=detrending_order,
                                        psd_freq_res=freq_band[0],
                                        psd_overlap=psd_overlap,
                                        psd_fs=fs,
                                        **analogsignal.annotations)
    return logMUA_asig

    # ToDo: Normalization with basline power?

def remove_duplicate_properties(objects, del_keys=['nix_name', 'neo_name']):
    if type(objects) != list:
        objects = [objects]
    for i in range(len(objects)):
        for k in del_keys:
            if k in objects[i].annotations:
                del objects[i].annotations[k]
    return None

if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--output",    nargs='?', type=str)
    CLI.add_argument("--data",      nargs='?', type=str)
    CLI.add_argument("--freq_band",  nargs=2, type=float)
    CLI.add_argument("--detrending_order", nargs='?', type=int, default=2)
    CLI.add_argument("--psd_overlap",  nargs='?', type=float)

    args = CLI.parse_args()

    with neo.NixIO(args.data) as io:
        segment = io.read_block().segments[0]

    remove_duplicate_properties([asig for asig in segment.analogsignals])

    logMUA_segment = build_logMUA_segment(segment,
                                          freq_band=args.freq_band,
                                          detrending_order=args.detrending_order,
                                          psd_overlap=args.psd_overlap)

    block = neo.core.Block(name='Results of {}'\
                                .format(os.path.basename(__file__)))
    logMUA_segment.description = 'Estimated logMUA activity'
    block.segments.append(logMUA_segment)
    with neo.NixIO(args.output) as io:
        io.write_block(block)