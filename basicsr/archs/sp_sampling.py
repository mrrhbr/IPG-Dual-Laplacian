import torch

def SP_global_sampling(x,
                      sp_map,
                      output = 0,
                      tp = 'bhwc',
                      return_inverse = False
    ):
    """
    Superpixel-based global sampling using mean pooling.

    Args : 
      x : Feature map.
      [B,H,W,C] if tp = 'bhwc'
      [B,C,H,W] if tp = 'bchw'


      sp_map:
        Superpixel labels , Shape [B,H,W].
        Labels shoud be non-negative integers. each label indicates superpixel id which the pixel belongs.

      output : 
        0 : return superpixel features (for Q)
        1 : return superpixel features (for K/V)
        2 : return (X_SP , X_SP) (for graph construction in calc_graph_)


      tp : 
        Feature map format : 'bhwc' or 'bchw'.

      return_inverse : 
        If True, also return the compact SP Index map.

    Returns : 
      X_SP : 
        [B,S,C]

      If output == 2 
        (X_SP,X_SP)

      If return_inverse:
        also returns inverse_map [B,H,W].
        
    """
