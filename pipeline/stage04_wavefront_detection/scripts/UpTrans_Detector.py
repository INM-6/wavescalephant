def ReadPixelData(events, DIM_X, DIM_Y, spatial_scale):

    import numpy as np
    from utils import remove_annotations
    import neo
    
    PixelLabel = events.array_annotations['y_coords'] * DIM_Y + events.array_annotations['x_coords']
    UpTrans = events.times
    Sorted_Idx = np.argsort(UpTrans)
    UpTrans = UpTrans[Sorted_Idx]
    PixelLabel = PixelLabel[Sorted_Idx]

    UpTrans_Evt = neo.Event(times=UpTrans,
                name='UpTrans',
                array_annotations={'channels':PixelLabel},
                description='Transitions from down to up states. '\
                           +'Annotated with the channel id ("channels")',
                Dim_x = DIM_X,
                Dim_y = DIM_Y,
                spatial_scale = spatial_scale)
    remove_annotations(UpTrans_Evt, del_keys=['nix_name', 'neo_name'])
    UpTrans_Evt.annotations.update(events.annotations)
    
    return(UpTrans_Evt)
