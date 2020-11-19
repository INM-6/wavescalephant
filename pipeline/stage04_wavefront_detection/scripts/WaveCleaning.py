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


def Neighbourhood_Search(h, w):
    
    #LOCALITY criteria
    neighbors = np.zeros([h*w, 4]);
    elements = np.array(range(0, h*w), dtype = 'int32')
    neighbors[:,0] = elements + 1 # right
    neighbors[:,1] = elements - 1 # left
    neighbors[:,2] = elements + w # down
    neighbors[:,3] = elements - w # up
    np.where(neighbors[:,0]>=w, neighbors[:,0], np.nan)
    np.where(neighbors[:,1]<0, neighbors[:,1], np.nan)
    np.where(neighbors[:,2]>h*w, neighbors[:,2], np.nan)
    np.where(neighbors[:,3]<0, neighbors[:,3], np.nan)

    return(neighbors)

def ChannelCleaning(Wave, neighbors):

    # --- (1) First CHANNEL CLEANING: check the TIME DISTANCE BETWEEN REPETITIONS IN A WAVE

    chs = Wave['ch'];            
    nhc_pixel, nhc_count = np.unique(chs, return_counts=True)
    rep = np.where(nhc_count > 1.)[0]; # channels index where the wave pass more than once           

    k = 0


    
    while k < len(rep):
        
        i=rep[k];
        
        repeted_index = np.where(chs == nhc_pixel[i])[0] # where i have repetitions
        
        timeStamp = Wave['times'][repeted_index].magnitude #UpTrans[idx];
        
        if np.max(np.diff(timeStamp))<0.125: #if repeted transition happen to be close in time                   
            # delta waves freqeunci is 1-4Hz -> minimum distance between two waves = 0.250s
            # open a time-window around each occurrence of i and check
            # how many members of the clan are in the time-window
            
            nClan=[];
            idx_noClan = []
            neigh_coll =  neighbors[nhc_pixel[i]]
            neigh_coll_idx = []
            for n in neigh_coll:
                neigh_coll_idx.extend(np.where(Wave['ch'] == n)[0])

            for idx, time in enumerate(timeStamp): # for each repetintion in the selected channel 
                time_window=[time-0.125, time+0.125];
                clan = len(np.intersect1d(Wave['times'][neigh_coll_idx] > time_window[0],
                                          Wave['times'][neigh_coll_idx] < time_window[1]))
                if clan == 0:
                    idx_noClan.append(idx)

            if len(idx_noClan)> 0:
                del_idx = repeted_index[idx_noClan]
                          
                del_idx = repeted_index[idx] # indexes to be deleted
                Wave['ndx'] = np.delete(Wave['ndx'], del_idx)
                Wave['times'] = np.delete(Wave['times'], del_idx)
                Wave['ch'] = np.delete(Wave['ch'], del_idx)
                Wave['WaveUnique'] = 1
                Wave['WaveTime'] = np.mean(Wave['times'])
                Wave['WaveSize'] = len(Wave['ndx'])

                chs = Wave['ch'];
                nhc_pixel, nhc_count = np.unique(chs, return_counts=True)
                if nhc_count[i] > 1:
                    k = k-1
        k = k+1
    return(Wave)

def Clean_SegmentedWave(seg, neighbors):

    delSeg=[];
               
    for i in range(0,len(seg)): # CHECK if SEGMENTS have to be cleaned
        
        # 1) clean non-unique segments
        if len(np.unique(seg[i]['ch'])) != len(seg[i]['ch']): #if  the subset of transition is NOT unique
            # --> 'CLEAN', i.e. scan the transition sequence and
            # find repetition of channels
        
            delCh = Clean_NonUniqueWave(seg[i], neighbors)
            
            if len(delCh):
                seg[i]['ch'] = np.delete(seg[i]['ch'], np.unique(delCh))
                seg[i]['ndx']= np.delete(seg[i]['ndx'], np.unique(delCh)); 
                seg[i]['times']= np.delete(seg[i]['times'], np.unique(delCh)); 


        # 2) clean channels non-LocallyConnected (see neighbors)                    
        if len(seg[i]['ch'])<=5: # 5 is min(len(neighbors{:}{2})
            delList=[];
            for ch_idx, ch in enumerate(seg[i]['ch']):
                if not (np.intersect1d(neighbors[ch], np.setdiff1d(seg[i]['ch'],ch))).size:
                    delList.append(ch_idx);
            if len(delList):
                seg[i]['ch']= np.delete( seg[i]['ch'],delList)
                seg[i]['ndx']= np.delete( seg[i]['ndx'],delList)
                seg[i]['times']= np.delete( seg[i]['times'],delList)

        # 3) prepere to remove empty segments
        if len(seg[i]['ndx'])==0:
            delSeg.append(i);


    # remove empty secments
    if delSeg:
       seg=np.delete(seg,delSeg);
    
    return(seg)


def Clean_NonUniqueWave(Wave, neighbors):              

    # 1) clean non-unique segments
    delCh=[];

    if len(np.unique(Wave['ch'])) != len(Wave['ch']): #if  the subset of transition is NOT unique
        # --> 'CLEAN', i.e. scan the transition sequence and
        # find repetition of channels
        
        delCh=[];

        involved_ch, repetition_count = np.unique(Wave['ch'], return_counts = True)
        involved_ch = involved_ch[np.where(repetition_count > 1)[0]] # channels non unique

        for rep_ch in involved_ch: # for each non unique channel delete repeted channels                                                  
            t0Clan=0;
            nClan=0;
            neigh = neighbors[rep_ch]
            
            for n in neigh:
                tClan = Wave['times'][np.where(Wave['ch']==n)[0]]
                if tClan.size > 0:
                    nClan=nClan+1; 
                    t0Clan=t0Clan+np.mean(tClan);
                
            if nClan > 0:
                t0Clan= np.float64(t0Clan)/nClan;
                tCh = Wave['times'][np.where(Wave['ch']==rep_ch)[0]]
                index = np.where(Wave['ch']==rep_ch)[0][np.argmin(abs(tCh-t0Clan))];                                
                delCh.extend(np.setdiff1d(np.where(Wave['ch']==rep_ch)[0], index));
                
    return(delCh)


def CleanWave(UpTrans,ChLabel, neighbors,  FullWave):

    FullWaveUnique=list(map(lambda x : x['WaveUnique'], FullWave))
    FullWaveSize=list(map(lambda x : x['WaveSize'], FullWave))
    FullWaveTime=list(map(lambda x : x['WaveTime'], FullWave))

    nPixel = len(np.unique(ChLabel))
    nw=0;

    
    while nw<len(FullWave): # for each wave
        

        if len(FullWave[nw]['ch']) != len(np.unique(FullWave[nw]['ch'])):
            
            # CLEAN wave channels
            FullWave[nw] = ChannelCleaning(FullWave[nw], neighbors)


            # Look for CANDIDATE SEGMENTS                            
            #Check the time difference between transitions
            delta=np.diff(FullWave[nw]['times']);
            mD=np.mean(delta);
            stdD=np.std(delta);
            THR=mD+3*stdD;

            Segments_Idx = np.where(delta>THR)[0] # where there is a larger sepration between transitions
            
            if Segments_Idx.size: # --- IDENTIFY SEGMENTS

                #create SEGMENTS i.e. candidate NEW waves
                n_candidate_waves=Segments_Idx.size +1; 
                istart=0;  i=0;

                seg = []
                for b in Segments_Idx:
                    seg.append({'ndx': list(range(istart,b+1)),
                                'ch': FullWave[nw]['ch'][istart:b+1],
                                'times': FullWave[nw]['times'][istart:b+1].magnitude});
                    istart=b+1;
                seg.append({'ndx': list(range(istart,len(FullWave[nw]['ch']))),
                            'ch': FullWave[nw]['ch'][istart:len(FullWave[nw]['ch'])],
                            'times': FullWave[nw]['times'][istart:len(FullWave[nw]['times'])].magnitude})

                
                # --- CLEAN SEGMENTS---
                seg = Clean_SegmentedWave(seg, neighbors)

                ##############################################################
                # fino a qui
                n_candidate_waves=len(seg); # update the value in 'segments' = number of segments

                # coalescence of segments if no repetitions with adjacent(s) one(s)
                # N.B. a "small" intersection is admitted
                i=0;
                while i<(n_candidate_waves-1): # for each wave
                    if len(np.intersect1d(seg[i]['ch'],seg[i+1]['ch'])) <= np.floor(1./4.*np.min([len(seg[i]['ch']),len(seg[i+1]['ch'])])):

                        # CANDIDATE SEGMENTS for COALESCENCE
                        # check also if distance between segments'border is smaller than 250ms = 1/4Hz

                        distance = seg[i+1]['times'][0] - seg[i]['times'][len(seg[i]['times'])-1];

                        if distance>=0.250: # check also if distance between segments'border is smaller than 250ms = 1/4Hz
                            # FREQUENCY ALERT: distance compatible with SWA, the two segments should be kept separated
                            i=i+1; #increment the pointer only if no coalescence is made
                        else:
                            # COALESCENCE
                            # The two segments are close enough that can be merged into a single wave
                            # (consider them separated waves would mean the SWA frequency is larger than 4Hz)
                            # COALESCENCE of consecutive SEGMENTS

                            merged = {'ch': np.append(seg[i]['ch'], seg[i+1]['ch']),
                                      'ndx': np.append(seg[i]['ndx'], seg[i+1]['ndx']),
                                      'times': np.append(seg[i]['times'], seg[i+1]['times'])}
                             
                            # CHECK for REPETITIONS (and treat them as usual...
                            # looking at the meanTime in the Clan)
                            
                            delCh = Clean_NonUniqueWave(merged, neighbors)

                            if len(np.unique(delCh))>0:
                                merged['ch'] = np.delete(merged['ch'], np.unique(delCh));
                                merged['ndx']= np.delete(merged['ndx'], np.unique(delCh));
                                merged['times']= np.delete(merged['times'], np.unique(delCh));

                            seg[i]=merged; # COALESCENCE
                            seg=np.delete(seg, i+1); # coalesced segments are at index i, segment at index i+1 is REMOVED
                            n_candidate_waves=n_candidate_waves-1;
                    else: # consecutive segments intersect too much...
                        i=i+1; #increment the pointer only if no coalescence is made

                    

                # $$$$$ N.B. the number of segments has to be updated
                NewFullWave = []
                for i in range(0,n_candidate_waves):
                    #ndx = []
                    #for elem in seg[i]['ndx']:
                    #    ndx.append(FullWave[nw]['ndx'][elem])
                    ndx = FullWave[nw]['ndx'][seg[i]['ndx']]
                    
                    NewFullWave.append({'ndx': ndx, 'times': UpTrans[ndx], 'ch': ChLabel[ndx],
                                        'WaveUnique':1, 'WaveSize': len(ndx), 'WaveTime': np.mean(UpTrans[ndx])});


                    #NewFullWaveUnique.append(1); # update the logical value (...we are "cleaning" the waves)
            else: # NO SEGMENTS identified -->
                # repeated chiannels are due to noise (and not to the presence of more than one wave)
                # CLEAN the wave, i.e. keep only the first channel occurrance
                delCh = Clean_NonUniqueWave(FullWave[nw], neighbors)
                
                if len(delCh):
                    FullWave[nw]['ch'] = np.delete(FullWave[nw]['ch'], np.unique(delCh))
                    FullWave[nw]['ndx']= np.delete(FullWave[nw]['ndx'], np.unique(delCh)); 
                    FullWave[nw]['times']= np.delete(FullWave[nw]['times'], np.unique(delCh)); 

                
                # wave is "cleaned"; store the updated wave
                ndx = FullWave[nw]['ndx']
                NewFullWave = {'ndx': ndx, 'times': UpTrans[ndx], 'ch': ChLabel[ndx],
                               'WaveUnique': 1, 'WaveSize': len(ndx), 'WaveTime':np.mean(UpTrans[ndx])};
                
            # --- REPLACE CurrentWave with NewWave(s)
            # [its segments or its 'cleaned' version]
            
            if nw != 0:
                Pre = FullWave[:].copy()
                FullWave=Pre[0:nw].copy()
                FullWave.extend(NewFullWave)
                FullWave.extend(Pre[nw+1:]);#end
            else:               
                Pre = FullWave[:].copy()
                FullWave = NewFullWave.copy()
                FullWave.extend(Pre[nw+1:]);#end
                
            # --- INCREMENT the pointer
            if len(NewFullWave)>0: # SEGMENTS ARE NEW WAVES
                nw = nw+len(NewFullWave); # increment (point at the next wave)
            else: # no segments identified, the current wave is a New Wave, because it has been cleaned
                nw=nw+1; # increment (point at the next wave)
        else:
            nw=nw+1; # increment (point at the next wave) [current wave is already unique]

    return(FullWave)


def RemoveSmallWaves(Evts_UpTrans, MIN_CH_NUM, FullWave):

    UpTrans = Evts_UpTrans.times
    ChLabel = Evts_UpTrans.array_annotations['channels']
    
    nCh = len(np.unique(ChLabel))
    DIM_X = Evts_UpTrans.annotations['Dim_x']
    DIM_Y = Evts_UpTrans.annotations['Dim_y']
    spatial_scale = Evts_UpTrans.annotations['spatial_scale']

    FullWaveUnique=list(map(lambda x : x['WaveUnique'], FullWave))
    FullWaveSize=list(map(lambda x : x['WaveSize'], FullWave))
    FullWaveTime=list(map(lambda x : x['WaveTime'], FullWave))


    ## Find Maxima and Minima in the histogram of FullWaveSize
 

    # Remove small waves and those rejected...
    temp = [FullWaveUnique[i]==1 and FullWaveSize[i] >= MIN_CH_NUM  for i in range(0, len(FullWaveUnique))]
    ndx = np.where(np.array(temp))[0];
    RejectedWaves = len(FullWaveUnique) - len(ndx); # rejected beacuse too small
                                                    # (i.e. involving too few channels)
    Wave = []
 
    for idx in ndx:
        Wave.append(FullWave[idx]);

    return Wave

    



