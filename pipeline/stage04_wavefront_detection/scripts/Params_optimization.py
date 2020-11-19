
def Optimal_MAX_ABS_TIMELAG(evts, MAX_ABS_TIMELAG):

    import numpy as np
    import quantities as pq
    from utils import load_neo, write_neo, remove_annotations
    import neo

    UpTrans = evts.times.magnitude
    ChLabel = evts.array_annotations['channels']
    DeltaTL = np.diff(UpTrans);

    ####################################################
    # Compute the time lags matrix looking for optimal MAX_ABS_TIMELAG...
    # (depends on the distance between electrodes in the array)

    WaveUnique_flag = 0;
    
    while WaveUnique_flag == 0: #while there still is at least one non-unique wave...
    
        WaveUnique_flag = 1
        WnW = np.int32(DeltaTL<=MAX_ABS_TIMELAG);
        ndxBegin_list = np.append(0, np.where(np.diff(WnW)==1)[0]+1)
        ndxEnd_list = np.append(np.where(np.diff(WnW)==-1)[0]+1, len(UpTrans)-1)

        Wave = [dict() for i in range(len(ndxBegin_list))]
        for w in range(0, len(ndxBegin_list)):
            
            ndx = list(range(ndxBegin_list[w],ndxEnd_list[w]+1))
            Wave[w] = {'ndx': ndx,
                       'ch': ChLabel[ndx],
                       'time': UpTrans[ndx],
                       'WaveUnique': len(ndx) == len(np.unique(ChLabel[ndx])),
                       'WaveSize': len(ndx),
                       'WaveTime': np.mean(UpTrans[ndx])};

            if len(ndx) != len(np.unique(ChLabel[ndx])):
                WaveUnique_flag = 0
                break
            
        MAX_ABS_TIMELAG = MAX_ABS_TIMELAG*0.75; # rescale the parameter
    print('max abs timelag', MAX_ABS_TIMELAG)
    
    return Wave

def Optima_MAX_IWI(UpTrans, ChLabel, Wave, ACCEPTABLE_REJECTION_RATE):

    import numpy as np
    import quantities as pq
    from utils import load_neo, write_neo, remove_annotations
    import neo

    WaveUnique=list(map(lambda x : x['WaveUnique'], Wave))
    WaveSize=list(map(lambda x : x['WaveSize'], Wave))
    WaveTime=list(map(lambda x : x['WaveTime'], Wave))

    nCh = len(np.unique(ChLabel))
    
    
    ####################################################
    # ExpectedTrans i.e. estimate of the Number of Waves
    # ExpectedTrans used to estimate/optimize IWI    
    TransPerCh_Idx, TransPerCh_Num = np.unique(ChLabel, return_counts=True)
    ExpectedTrans = np.median(TransPerCh_Num[np.where(TransPerCh_Num != 0)]);
    
    ## Wave Hunt -- step 1: Compute the Time Lags, Find Unique Waves --> WaveCollection1
    IWI = np.diff(WaveTime);          # IWI = Inter-Wave-Interval...


    ##Recollect small-waves in full waves (involving a wider area).
    MAX_IWI = np.max(IWI)*0.5
    OneMoreLoop = 1;
    
    while OneMoreLoop:

        WnW = np.int32(IWI<=MAX_IWI);
        ndxBegin_list = np.append(0, np.where(np.diff(WnW)==1)[0]+1)
        ndxEnd_list = np.append(np.where(np.diff(WnW)==-1)[0]+1, len(WaveTime)-1)

        FullWave = []

        num_iter = -1
        for w in range(0, len(ndxBegin_list)):
            num_iter = num_iter +1
            ndx = list(range(ndxBegin_list[w],ndxEnd_list[w]+1))
            Full_ndx = []
            for elem in ndx:
                Full_ndx.extend(Wave[elem]['ndx'])
            Full_ndx = np.int64(Full_ndx)
            
            FullWave.append({'ndx': Full_ndx,
                           'times': UpTrans[Full_ndx],
                           'ch': ChLabel[Full_ndx],
                           'WaveUnique': len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])),
                           'WaveSize': len(Full_ndx),
                           'WaveTime': np.mean(UpTrans[Full_ndx]),
                           'WaveUniqueSize': len(np.unique(ChLabel[Full_ndx]))
                           });

            # save as single waves isolated transitions.

            
            if ndxEnd_list[w]+1 != len(WaveTime):
                for j in range(ndxEnd_list[w]+1, ndxBegin_list[w+1]):
                    ndx = [j]
                    Full_ndx = []
                    for elem in ndx:
                        Full_ndx.extend(Wave[elem]['ndx'])
                    Full_ndx = np.int64(Full_ndx)
                    
                    FullWave.append({'ndx': Full_ndx,
                                   'times': UpTrans[Full_ndx],
                                   'ch': ChLabel[Full_ndx],
                                   'WaveUnique': len(Full_ndx) == len(np.unique(ChLabel[Full_ndx])),
                                   'WaveSize': len(Full_ndx),
                                   'WaveTime': np.mean(UpTrans[Full_ndx]),
                                   'WaveUniqueSize': len(np.unique(ChLabel[Full_ndx]))
                                   });

            
        FullWaveUnique = list(map(lambda x : x['WaveUnique'], FullWave))
        BadWavesNum = len(FullWaveUnique) - len(np.where(FullWaveUnique)[0]);
        OneMoreLoop = 0;

        if len(FullWaveUnique) <= ExpectedTrans: # If not we have an artifactual amplification of small waves...
            if float(BadWavesNum)/len(FullWaveUnique) > ACCEPTABLE_REJECTION_RATE:
                if np.min(FullWaveUnique) == 0: # at lest a Wave non-unique
                    MAX_IWI = MAX_IWI*0.75;
                    OneMoreLoop = 1;
                else: # only unique waves
                    if np.max(WaveSize) < nCh: # at least one wave MUST BE GLOBAL (i.e. involving the full set of electrodes)
                        MAX_IWI = MAX_IWI*1.25;
                        OneMoreLoop = 1;
    return(FullWave)





