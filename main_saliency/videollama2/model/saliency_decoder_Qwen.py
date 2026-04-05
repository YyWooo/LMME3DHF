import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import einops
from videotransformer import ViViTBackbone
import torch.nn.init as init
import torch.nn.functional as F

class Upsample(nn.Module):
    '''
    steal from Restormer
    '''
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2)) # 通道*2/4=/2， 尺寸*2

    def forward(self, x):
        return self.body(x)
    
class Upsample_decode(nn.Module):
    '''
    steal from Restormer
    '''
    def __init__(self, n_feat):
        super(Upsample_decode, self).__init__()

        self.upsample = Upsample(n_feat) # 通道/2， 尺寸*2
        self.reduce_chan = nn.Conv2d(n_feat, n_feat//2, kernel_size=1, bias=False)
        # self.conv = nn.Conv2d(n_feat//2, n_feat//2, kernel_size=3, padding=1, bias=False)

    def forward(self, x, x_skip):
        x = self.upsample(x) # 通道/2， 尺寸*2
        x = torch.cat([x_skip, x], dim=1) # 通道*2
        x = self.reduce_chan(x) # 通道/2
        # x = self.conv(x)
        return x # 总结： 通道/2，尺寸*2
    
class temporal_GRU(nn.Module):
    def __init__(self, shape_CTHW):
        super().__init__()

        self.shape_CTHW = shape_CTHW
        C, T, H, W = self.shape_CTHW
        hidden_size = C 
        # self.proj1 = nn.Conv3d(C, C//20, kernel_size=1, bias=False)
        # self.proj2 = nn.Conv3d(C//20, C, kernel_size=1, bias=False)
        self.gru_spatial = nn.GRU(input_size=C, hidden_size=hidden_size)
        self.gru_temporal = nn.GRU(input_size=C, hidden_size=hidden_size)

    def forward(self, x): # x [b,c,t,h,w] torch.Size([1, 1152, 8, 13, 13])
        # x = self.proj1(x) 
        x = einops.rearrange(x, 'b c t h w -> (h w) (b t) c')  
        out_spa, hn_spa = self.gru_spatial(x)
        x = einops.rearrange(out_spa, 'hw (b t) c -> t (b hw) c', b=1)  
        _, hn_tem = self.gru_temporal(x) # 1 (1 hw) c
        # last_hidden_state = hn_tem[-1]  # 取 hn 的最后一个层 [1,1,c*h*w]

        C, T, H, W = self.shape_CTHW
        output = einops.rearrange(hn_tem, '1 (h w) c -> 1 c 1 h w', c=C, h=H, w=W)  # [1, C, H, W]
        # output = self.proj2(output)
        return output
    
class SaliencyDecoder(nn.Module):
    def __init__(self, feature_dim=3584):
        super(SaliencyDecoder, self).__init__()
        
        self.relu = nn.ReLU()
        self.reduce_dim_linear1 = nn.Linear(in_features=3584, out_features=3584, bias=True)
        self.reduce_dim_linear2 = nn.Linear(in_features=3584, out_features=1152, bias=True)
        self.conv3d = nn.Conv3d(in_channels=1152, 
                                 out_channels=1152, 
                                 kernel_size=(8, 3, 3), 
                                 stride=[1,1,1], 
                                 padding=[0,1,1], 
                                 bias=True) 
  
        self.deconv1 = nn.ConvTranspose2d(in_channels=1152, out_channels=576, kernel_size=3, stride=2, padding=0,  output_padding=0)
        self.upsample1 = Upsample(576)
        self.upsample2 = Upsample(288)
        self.upsample3 = Upsample(144)
        
        self.vision_reduce_dim1 = nn.Conv2d(1152, 576, kernel_size=1, stride=1, padding=0)
        self.vision_reduce_dim2 = nn.Sequential(
            nn.Conv2d(1152, 576, kernel_size=1, stride=1, padding=0),
            Upsample(576)
        )
        self.vision_reduce_dim3 = nn.Sequential(
            nn.Conv2d(1152, 576, kernel_size=1, stride=1, padding=0),
            Upsample(576),
            Upsample(288),
            )

        self.decode_reduce_dim1 = nn.Conv2d(1152, 576, kernel_size=1, stride=1, padding=0)
        self.decode_reduce_dim2 = nn.Conv2d(576, 288, kernel_size=1, stride=1, padding=0)
        self.decode_reduce_dim3 = nn.Conv2d(288, 144, kernel_size=1, stride=1, padding=0)
        
        self.refine_conv1 = nn.Conv2d(72, 36, kernel_size=3, stride=1, padding=1)
        self.refine_conv2 = nn.Conv2d(36, 18, kernel_size=3, stride=1, padding=1)
        self.refine_conv3 = nn.Conv2d(18, 1, kernel_size=1)  # 输出单通道的显著性图

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights()
        
    def init_weights(self):
        # Xavier Uniform 初始化方法
        init.xavier_uniform_(self.reduce_dim_linear1.weight)
        init.xavier_uniform_(self.reduce_dim_linear2.weight)

        # 偏置初始化为零
        init.constant_(self.reduce_dim_linear1.bias, 0)
        init.constant_(self.reduce_dim_linear2.bias, 0)    
        
    def forward(self, x, vision_hidden_states): # [b, 1352, 3584]与[16,27*27,1152]
        # x = x.to(torch.float32)
        v_2 = vision_hidden_states[0]
        # v_2 = v_2.to(torch.float32)
        v_12 = vision_hidden_states[1]
        # v_12 = v_12.to(torch.float32)
        v_22 = vision_hidden_states[2]
        # v_22 = v_22.to(torch.float32)
        
        x = self.reduce_dim_linear1(x) # ->[b,1352,3584]
        x = self.dropout(x)
        x = self.reduce_dim_linear2(x) # ->[b,1352,1152]
        x = einops.rearrange(x, 'b (t h w) d -> b d t h w', t=8, h=13, w=13) # -> [b,1152,8,13,13]
        x = self.conv3d(x).squeeze(2) # -> [b,1152,13,13]
        
        v_2 = einops.rearrange(v_2[8], '(h w) d -> 1 d h w', h=27, w=27) # -> [b,1152,27,27]
        v_12 = einops.rearrange(v_12[8], '(h w) d -> 1 d h w', h=27, w=27) # -> [b,1152,27,27]
        v_22 = einops.rearrange(v_22[8], '(h w) d -> 1 d h w', h=27, w=27) # -> [b,1152,27,27]
        
        x = self.deconv1(x) # -> [b,576,27,27]
        v_2 = self.vision_reduce_dim1(v_2) # ->[b,576,27,27]
        
        x = torch.cat([x,v_2], dim=1) # -> [b,1152,27,27]
        x = self.decode_reduce_dim1(x) # -> [b,576,27,27]
        x = self.dropout(x)
        x = self.upsample1(x) # -> [b,288,54,54]
        
        v_12 = self.vision_reduce_dim2(v_12) # -> [b, 288,54,54]
        
        x = torch.cat([x,v_12], dim=1) # -> [b,576,54,54]
        x = self.decode_reduce_dim2(x) # -> [b,288,54,54]
        x = self.dropout(x)
        x = self.upsample2(x) # -> [b,144,108,108]
        
        v_22 = self.vision_reduce_dim3(v_22) # -> [b,144,108,108]
        
        x = torch.cat([x,v_22], dim=1) # -> [b,288,108,108]
        x = self.decode_reduce_dim3(x) # -> [b,144,108,108]
        x = self.dropout(x)
        x = self.upsample3(x) # -> [b,72,216,216]
        
        x = self.refine_conv1(x)
        x = self.refine_conv2(x)
        x = self.refine_conv3(x)
        
        x = x.squeeze(1)
        x = (self.tanh(x) + 1) / 2 

        return x  # The final output is the saliency map with the same size as the input image
