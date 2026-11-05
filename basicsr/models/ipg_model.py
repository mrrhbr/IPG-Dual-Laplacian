import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class IPGModel(SRModel):
    
    def test(self):
        # Pad to multiplication of sample_size
        sample_size = self.opt['network_g']['sample_size']
        scale = self.opt.get('scale', 1)
    
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
    
        if h % sample_size != 0:
            mod_pad_h = sample_size - h % sample_size
    
        if w % sample_size != 0:
            mod_pad_w = sample_size - w % sample_size
    
        img = F.pad(
            self.lq,
            (0, mod_pad_w, 0, mod_pad_h),
            'reflect'
        )
    
        sp_map = self.sp_map
    
        if mod_pad_h > 0 or mod_pad_w > 0:
            sp_map = F.pad(
                sp_map.unsqueeze(1).float(),
                (0, mod_pad_w, 0, mod_pad_h),
                mode='replicate'
            ).squeeze(1).long()
    
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(
                    img,
                    sp_map=sp_map
                )
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(
                    img,
                    sp_map=sp_map
                )
            self.net_g.train()
    
        _, _, h, w = self.output.size()
    
        self.output = self.output[
            :, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
