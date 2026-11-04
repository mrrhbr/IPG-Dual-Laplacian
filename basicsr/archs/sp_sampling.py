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

    if tp == 'bchw':
        x = x.permute(0,2,3,1)
    
    elif tp != 'bhwc':
      raise ValueError(f'Unsupported tensor format : {tp}')
    
    B,H,W,C = x.shape
    
    if sp_map.shape != (B,H,W):
      raise ValueError(f'Expected sp_map shape {B,H,W}',
                      f'got {tuple(sp_map.shape)}')
    
    sp_map = sp_map.to(device = x.device , dtype = torch.long)
    
    if torch.any(sp_map < 0 ):
      raise ValueError('Superpixel IDs must be non_negative.')
    
    inverse_map = torch.empty_like(sp_map)
    sp_ids = []
    max_sp = 0
    
    for b in range(B):
      unique_ids , inverse = torch.unique(
        sp_map[b].reshape(-1),
        sorted = True,
        return_inverse = True
      )
      inverse_map[b] = inverse.reshape(H,W)
      sp_ids.append(unique_ids)
      max_sp = max(max_sp , unique_ids.numel())
    
    X_SP = x.new_zoes(B,max_sp,C)
    
    for b in range(B):
      labels = inverse_map[b].reshape(-1)
      features = x[b].reshape(H*W,C)
    
      sums = x.new_zeros(max_sp , C)
      sums.index_add_(0,labels,features)
    
      counts = torch.bincount(
        labels,
        minlength = max_sp).to(dtype = x.dtype)
    
      X_SP[b] = sums / count.clamp_min(1).unsqueeze(-1)
    
    if output not in (0,1,2):
      raise ValueError(f'Unsupported output :{output}')
    if output == 2 :
      result = (X_SP , X_SP)
      
    else:
      result = X_SP
      
    if return_inverse :
      return( *result , inverse_map ) if isinstance(result, tuple) else(
        result , inverse_map
      )
    
    return result


def SP_broadcast (x_sp , inverse_map):

  """
  X_SP  = [B,S,C]
  inverse_map = [B,H,W]
  """
  B, H, W  = inverse_map.shape
  C  = x_sp.shape[-1]

  x_out = torch.gather(
    x_sp,
    dim = 1,
    index = index.unsqueeze(-1).expand(-1,-1,C)
  )
  return x_out.reshape(B,H,W,C)
  
  
