import os
import numpy as np
import quantities as pq
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import io
import math
from utils import AnalogSignal2ImageSequence, load_neo, save_plot
from utils import load_neo, write_neo, remove_annotations
from utils import none_or_str, none_or_float
import neo

from UpTrans_Detector import ReadPixelData
from Params_optimization import Optimal_MAX_ABS_TIMELAG, Optima_MAX_IWI
#from Optimal_MAX_ABS_TIMELAG import Optimal_MAX_ABS_TIMELAG
#from MAX_IWI_Search import MAX_IWI_Search
#from WaveHuntTimes_UnicityPrinciple import WaveHuntTimes_UnicityPrinciple, Neighbourhood_Search
#from WaveHuntTimes_Globality import WaveHuntTimes_Globality
from WaveCleaning import RemoveSmallWaves, CleanWave, Neighbourhood_Search
# ======================================================================================#

# LOAD input data
if __name__ == '__main__':
    
    CLI = argparse.ArgumentParser(description=__doc__,
                      formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--data", nargs='?', type=str, required=True,
                        help="path to input data in neo format")
    CLI.add_argument("--output", nargs='?', type=str, required=True,
                        help="path of output file")
    CLI.add_argument("--Max_Abs_Timelag", nargs='?', type=float, default=0.8,
                        help="Maximum reasonable time lag between electrodes (pixels)")
    CLI.add_argument("--Acceptable_rejection_rate", nargs='?', type=float, default=0.1,
                        help=" ")
    CLI.add_argument("--min_ch_num", nargs='?', type=float, default=300,
                        help="minimum number of channels involved in a wave")
  
    args = CLI.parse_args()
    block = load_neo(args.data)
    block = AnalogSignal2ImageSequence(block)
    imgseq = block.segments[0].imagesequences[-1]
    
    DIM_X, DIM_Y = np.shape(imgseq[0])
    spatial_scale = imgseq.spatial_scale
    
    asig = block.segments[0].analogsignals[-1]
    evts = [ev for ev in block.segments[0].events if ev.name== 'Transitions']
    if len(evts):
        evts = evts[0]
    else:
        raise InputError("The input file does not contain any 'Transitions' events!")
    
    # Transform the ouptut of stage 3in a more suitable mode
    UpTrans_Evt = ReadPixelData(evts, DIM_X, DIM_Y, spatial_scale)

    # search for the optimal abs timelag
    Waves_Inter = Optimal_MAX_ABS_TIMELAG(UpTrans_Evt, args.Max_Abs_Timelag)
    
    # search for the best max_IWI param
    Waves_Inter = Optima_MAX_IWI(UpTrans_Evt.times, UpTrans_Evt.array_annotations['channels'], Waves_Inter, args.Acceptable_rejection_rate)

    # Unicity principle refinement
    neighbors = Neighbourhood_Search(UpTrans_Evt.annotations['Dim_x'], UpTrans_Evt.annotations['Dim_y'])
    Waves_Inter = CleanWave(UpTrans_Evt.times, UpTrans_Evt.array_annotations['channels'], neighbors, Waves_Inter)

    # Globality principle
    Wave = RemoveSmallWaves(UpTrans_Evt, args.min_ch_num, Waves_Inter)


        # Create an array with the beginning time of each wave
    # Saves the array in a 'BeginTime.txt' file in the path folder

    Waves = []
    Times = []
    Label = []
    Pixels = []
    
    for i in range(0,len(Wave)):
        Times.extend(Wave[i]['times'])
        Label.extend(np.ones([len(Wave[i]['ndx'])])*i)
        Pixels.extend(Wave[i]['ch'])

    Times = Times*(Wave[0]['times'].units)
    waves = neo.Event(times=Times.rescale(pq.s),
                    labels=np.int64(Label),
                    name='Wavefronts',
                    array_annotations={'channels':Pixels,
                                       'x_coords':[p % DIM_Y for p in Pixels],
                                       'y_coords':[np.floor(p/DIM_Y) for p in Pixels]},
                    description='Transitions from down to up states. '\
                               +'Labels are ids of wavefronts. '
                               +'Annotated with the channel id ("channels") and '\
                               +'its position ("x_coords", "y_coords").',
                    spatial_scale = UpTrans_Evt.annotations['spatial_scale'])

    #remove_annotations(waves, del_keys=['nix_name', 'neo_name'])
    waves.annotations.update(evts.annotations)
    remove_annotations(waves, del_keys=['nix_name', 'neo_name'])

    block.segments[0].events.append(waves)
    write_neo(args.output, block)
