import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fdf.thetaconv.activations import SnakeBeta, Activation1d
import torch.nn.functional as F
from torch_gfrft.torch_gfrft.gft import GFT
from torch_gfrft.torch_gfrft.gfrft import GFRFT
from torch_gfrft.torch_gfrft.layer import GFTLayer, IGFTLayer, GFRFTLayer
from torch_gfrft.torch_gfrft import EigvalSortStrategy, ComplexSortStrategy



def frft_ozaktas(
    x: torch.Tensor,
    alphas: torch.Tensor,
    time_dim: int = -2,
    feature_dim: int = -1,
) -> torch.Tensor:
   

   
    if feature_dim < 0:
        feature_dim += x.dim()
    if time_dim < 0:
        time_dim += x.dim()

    permute_order = [d for d in range(x.dim()) if d not in (time_dim, feature_dim)]
    permute_order += [time_dim, feature_dim]           
    x = x.permute(*permute_order)
    T, F = x.shape[-2], x.shape[-1]
    batch_shape = x.shape[:-2]

    device = x.device
    x = x.to(torch.complex64)

  
    alphas = alphas.to(device=device, dtype=torch.float32).reshape(
        *([1] * len(batch_shape)), F
    )  

    phi = alphas * (math.pi / 2.0)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_phi_safe = torch.where(sin_phi.abs() < 1e-8, torch.ones_like(sin_phi), sin_phi)
    cot_phi = cos_phi / sin_phi_safe
    csc_phi = 1.0 / sin_phi_safe

    
    t = torch.arange(T, device=device, dtype=torch.float32) - (T - 1) / 2.0
    t2 = (t ** 2).view(*([1] * len(batch_shape)), T, 1)  
   
    pre_chirp = torch.exp(-1j * math.pi * cot_phi * t2)  
    x_pre = x * pre_chirp

   
    L = 2 * T
    n = torch.arange(-T, T, device=device, dtype=torch.float32)
    n2 = (n ** 2).view(*([1] * len(batch_shape)), L, 1)   # (..., L, 1)
    g = torch.exp(1j * math.pi * csc_phi * n2)            # (..., L, F)

    X_fft = torch.fft.fft(x_pre, n=L, dim=-2)
    G_fft = torch.fft.fft(g,      dim=-2)
    conv = torch.fft.ifft(X_fft * G_fft, dim=-2)

    
    start = T - 1
    conv = conv[..., start:start + T, :]

    
    post_chirp = torch.exp(-1j * math.pi * cot_phi * t2)
    scale = torch.exp(-1j * (1 - alphas) * math.pi / 4.0) / torch.sqrt(torch.abs(sin_phi_safe))
    y = scale * post_chirp * conv

    
    eps = 1e-8
    need_exact = ((alphas.abs() < eps) |
                  ((alphas - 1).abs() < eps) |
                  ((alphas + 1).abs() < eps) |
                  ((alphas - 2).abs() < eps))

    if need_exact.any():
        # broadcast masks
        mask_id   = (alphas.abs() < eps).expand_as(y)
        mask_fft  = ((alphas - 1).abs() < eps).expand_as(y)
        mask_ifft = ((alphas + 1).abs() < eps).expand_as(y)
        mask_rev  = ((alphas - 2).abs() < eps).expand_as(y)

        y = torch.where(mask_id,   x,                          y)
        y = torch.where(mask_fft,  torch.fft.fft(x,  dim=-2),  y)
        y = torch.where(mask_ifft, torch.fft.ifft(x, dim=-2),  y)
        y = torch.where(mask_rev,  torch.flip(x, dims=[-2]),   y)

    
    inv_perm = [0] * x.dim()
    for i, p in enumerate(permute_order):
        inv_perm[p] = i
    y = y.permute(*inv_perm).contiguous()
    return y

class Sin(nn.Module):
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)

class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, lr_pos_emb: float = 1e-5, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]
        
        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=True)
        self.t = nn.Parameter(t, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]

class ExponentialModulation(nn.Module):
    def __init__(
            self,
            d_model,
            fast_decay_pct=0.3,
            slow_decay_pct=1.5,
            target=2e-2,
            modulation_lr=0.0,
            modulate: bool = True,
            shift: float = 0.0,
            **kwargs
    ):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.deltas = nn.Parameter(deltas, requires_grad=False)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x

class Filter(nn.Module):
    def __init__(
            self,
            d_model,
            emb_dim=3,
            order=16,
            fused_fft_conv=False,
            seq_len=1024,
            lr=1e-3,
            lr_pos_emb=1e-5,
            dropout=0.0,
            w=1,
            wd=0,
            bias=True,
            num_inner_mlps=2,
            normalized=False,
            **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_bias = bias
        self.fused_fft_conv = fused_fft_conv
        self.bias = nn.Parameter(torch.randn(self.d_model))
        self.dropout = nn.Dropout(dropout)

        act = Sin(dim=order, w=w)
        self.emb_dim = emb_dim
        assert emb_dim % 2 != 0 and emb_dim >= 3, "emb_dim must be odd and greater or equal to 3"
        self.seq_len = seq_len

        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, lr_pos_emb)

        self.implicit_filter = nn.Sequential(
            nn.Linear(emb_dim, order),
            act,
        )
        for i in range(num_inner_mlps):
            self.implicit_filter.append(nn.Linear(order, order))
            self.implicit_filter.append(act)

        self.implicit_filter.append(nn.Linear(order, d_model, bias=False))

        self.modulation = ExponentialModulation(d_model, **kwargs)
        self.normalized = normalized

    def _filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)
        return h

class DSWEmbedding(nn.Module):
    def __init__(self, seg_len, d_model, max_channels=256):
        super().__init__()
        self.seg_len = seg_len
        self.embed = nn.Sequential(nn.Conv1d(1, d_model, kernel_size=seg_len, stride=seg_len // 4),
                                   nn.InstanceNorm1d(d_model, affine=True, eps=1e-6),
                                   Activation1d(SnakeBeta(d_model)),
                                   nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm1d(d_model, affine=True, eps=1e-6),
                                   Activation1d(SnakeBeta(d_model)),
                                   nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1),
                                   nn.InstanceNorm1d(d_model, eps=1e-6, affine=True))
        self.channel_embed = nn.Conv2d(d_model, d_model, kernel_size=(8, 3), stride=(4, 1), padding=(0, 1))
        self.pos_embed = nn.Parameter(torch.randn(1, max_channels, d_model))

    def forward(self, x):
        batch_size, timesteps, nvars = x.size()
        x = rearrange(x, "b l n -> (b n) l")[:, None, :]
        x = self.embed(x)
        x = rearrange(x, "(b n) d seg_num -> b n seg_num d", n=nvars, b=batch_size)
        x = self.channel_embed(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        return x

class ConvMLPBlock(nn.Module):
    def __init__(self, d_model, expansion=4, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.GELU()
        )
        
        self.conv_norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        
       
        if x.dim() == 3 and x.size(1) > 1: 
            
            x_conv = x.permute(0, 2, 1)
            conv_out = self.conv(x_conv)
            
            conv_out = conv_out.permute(0, 2, 1)
            
            conv_out = self.conv_norm(conv_out)
        else:
            
            conv_out = 0
        
        mlp_out = self.mlp(x)
        out = conv_out + mlp_out + residual
        return self.norm(out)

class FMOperatorLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            l_max: int = 1024,
            order: int = 2,
            filter_order: int = 64,
            dropout: float = 0.1,
            filter_dropout: float = 0.0,
            static: bool = True,
            num_heads: int = 4,
            **filter_args,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.l_max = l_max
        self.order = order
        self.dropout = nn.Dropout(dropout)
        self.static = static
        
        inner_width = int(embed_dim * (1 + order))
        
        self.in_proj = nn.Linear(embed_dim, inner_width)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.GELU()
        self.alpha = nn.Parameter(0.5 * torch.randn(embed_dim))
        self.short_filter = nn.Conv1d(inner_width, inner_width, 3, padding=1, groups=inner_width)
        
        if self.static:
            self.filter_fn = Filter(embed_dim * (order - 1),
                                    order=filter_order,
                                    seq_len=l_max,
                                    channels=1,
                                    dropout=filter_dropout,
                                    **filter_args)
        else:
            self.ctx_proj = nn.Linear(embed_dim, embed_dim)
            self.filter_fn = DynamicFilter(embed_dim * (order - 1),
                                           order=filter_order,
                                           seq_len=l_max,
                                           channels=1,
                                           dropout=filter_dropout,
                                           **filter_args)

        self.num_heads = num_heads
        self.norm_scale = nn.Parameter(torch.ones(embed_dim // num_heads))
        self.norm_params = nn.Parameter(torch.randn(num_heads, num_heads))

    def _multihead_norm(self, x):
        x = rearrange(x, "b l (n d) -> b l d n", n=self.num_heads)
        std = torch.std(x, dim=-2, keepdim=True) + 1e-5
        x = (self.norm_scale[:, None] * x / std).matmul(self.norm_params)
        x = rearrange(x, "b l d n -> b l (n d)")
        return x

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        dtype = x.dtype
        bsz, tgt_len, _ = x.size()
        l_filter = min(tgt_len, self.l_max)
        
        u = self.in_proj(x).transpose(-2, -1)
        u = self.activation(u)
        uc = self.short_filter(u)
        
        original_x = x
        *split_parts, v = uc.split(self.embed_dim, dim=1)
        
        if self.static:
            k = self.filter_fn._filter(l_filter).squeeze(0)
            k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        else:
            if context is None:
                context = original_x
            k = self.filter_fn._filter(l_filter, self.ctx_proj(context))
            k = rearrange(k, 'b l (o d) -> o b d l', o=self.order - 1)

        alpha = self.alpha.sigmoid()
        
        for o, x_i in enumerate(reversed(split_parts[1:])):
            v = self.dropout(v * x_i)
            k = k[o]
            k, v = k.transpose(-2, -1), v.transpose(-2, -1)
            v_frft = frft_ozaktas(v, alpha)
            
            if k.ndim < v_frft.ndim:
                k = k[None, :, :]
                
            v_frft = v_frft * frft_ozaktas(k, alpha)
            v = frft_ozaktas(v_frft, -alpha).real
        
        y = self._multihead_norm(v) * split_parts[0].sigmoid().transpose(-2, -1)
        y = self.out_proj(y)
        return y.contiguous()



class ComplexToReal(nn.Module):
   
    def forward(self, x):
        return torch.abs(x)

class GraphFeatureExtractor(nn.Module):
    
    def __init__(self, config, is_gc=False):
        super().__init__()
        self.config = config
        self.is_gc = is_gc
        
        self.node_count = config.enc_in
        
        
        self.gfrft_layers = nn.ModuleList()
        
        num_layers = config.freq_bands if is_gc else 1
        for _ in range(num_layers):
            self.gfrft_layers.append(nn.Sequential(
                GFRFTLayer(
                    GFRFT(torch.eye(self.node_count)),
                    order=0.5,
                    trainable=True
                ),
                ComplexToReal(),
                nn.Linear(self.node_count, self.node_count)
            ))
        
        self.fc1 = nn.Linear(self.node_count, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.bn = nn.BatchNorm1d(config.hidden_dim // 2)
        self.dropout = nn.Dropout(config.dropout)
        
        self.out_dim = config.hidden_dim // 2 * (config.freq_bands if is_gc else 1)

    def forward(self, x):
        if self.is_gc:
            if x.dim() == 4:
                x = x.unsqueeze(2)
            elif x.dim() > 5:
                while x.dim() > 5:
                    x = x.squeeze(1)
            
            batch, windows, bands, nodes, _ = x.shape
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)
            elif x.dim() == 5:
                while x.dim() > 4:
                    x = x.squeeze(1)
            elif x.dim() != 4:
                raise ValueError(f"Unexpected input dimension for wPLI graph: {x.dim()}")
            
            batch, windows, nodes, _ = x.shape
        
        features = []
        
        if self.is_gc:
            x = x.view(batch * windows, bands, nodes, nodes)
            
            for band in range(bands):
                graph = x[:, band, :, :]
                
                layer_outputs = []
                for i in range(graph.shape[0]):
                    gft = GFT(
                        graph[i],
                        eigval_sort_strategy=self.config.eigval_sort_strategy,
                        complex_sort_strategy=self.config.complex_sort_strategy
                    )
                    gfrft = GFRFT(gft.gft_mtx)
                    
                    self.gfrft_layers[band][0].gfrft = gfrft
                    
                    node_feats = torch.eye(nodes, device=graph.device)
                    gfrft_out = self.gfrft_layers[band](node_feats)
                    layer_outputs.append(gfrft_out)
                
                band_feat = torch.stack(layer_outputs).mean(dim=1)
                band_feat = F.relu(self.bn(self.fc2(F.relu(self.fc1(band_feat)))))
                band_feat = self.dropout(band_feat)
                features.append(band_feat)
            
            x = torch.cat(features, dim=1)
            x = x.view(batch, windows, -1)
            x = x.mean(dim=1)
            
        else:
            x = x.view(batch * windows, nodes, nodes)
            
            layer_outputs = []
            for i in range(x.shape[0]):
                gft = GFT(
                    x[i],
                    eigval_sort_strategy=self.config.eigval_sort_strategy,
                    complex_sort_strategy=self.config.complex_sort_strategy
                )
                gfrft = GFRFT(gft.gft_mtx)
                
                self.gfrft_layers[0][0].gfrft = gfrft
                
                node_feats = torch.eye(nodes, device=x.device)
                gfrft_out = self.gfrft_layers[0](node_feats)
                layer_outputs.append(gfrft_out)
            
            x = torch.stack(layer_outputs).mean(dim=1)
            x = F.relu(self.bn(self.fc2(F.relu(self.fc1(x)))))
            x = self.dropout(x)
            
            x = x.view(batch, windows, -1)
            x = x.mean(dim=1)
        
        return x



class ModelConfig:
    def __init__(self, enc_in=64, n_samples=2048, cache_dir=None, task_name='classification'):
        self.enc_in = enc_in
        self.n_samples = n_samples
        self.cache_dir = cache_dir
        self.task_name = task_name
        
        self.eigval_sort_strategy = EigvalSortStrategy.TOTAL_VARIATION
        self.complex_sort_strategy = ComplexSortStrategy.ABS_ANGLE_02pi
        
        self.hidden_dim = 64
        self.num_heads = 4
        self.depth = 4
        self.dropout = 0.2
        self.num_classes = 2
        self.freq_bands = 1

class FeatureFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
       
        self.channel1_embed = DSWEmbedding(seg_len=128, d_model=config.hidden_dim, max_channels=config.enc_in)
        
        self.channel1_layers = nn.ModuleList([
            FMOperatorLayer(
                embed_dim=config.hidden_dim,
                l_max=2048,
                order=2,
                filter_order=32,
                static=True,
                num_heads=4
            ) for _ in range(4)
        ])
        
        self.channel1_mlp_blocks = nn.ModuleList([
            ConvMLPBlock(config.hidden_dim) for _ in range(4)
        ])
        
        self.channel1_norm = nn.LayerNorm(config.hidden_dim)
        
       
        self.wpli_extractor = GraphFeatureExtractor(config, is_gc=False)
        self.gc_extractor = GraphFeatureExtractor(config, is_gc=True)
        
       
        in_features = config.hidden_dim + self.wpli_extractor.out_dim + self.gc_extractor.out_dim
        out_features = config.hidden_dim * 2
        
        print(f"Fusion layer input features: {in_features}, output features: {out_features}")
        
    
        self.fusion_proj = nn.Linear(in_features, out_features)
        self.fusion_norm = nn.LayerNorm(out_features)
        self.fusion_dropout = nn.Dropout(config.dropout)
        
       
        self.conv_mlp_blocks = nn.ModuleList([
            ConvMLPBlock(out_features) for _ in range(2)
        ])
        
       
        if config.task_name == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(out_features, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(config.hidden_dim, config.num_classes)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(out_features, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(config.hidden_dim, 1)
            )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, eeg_data, wpli_graphs, gc_graphs):
        
        x1 = self.channel1_embed(eeg_data)  
        batch_size, n, seg_num, d_model = x1.shape
        x1 = rearrange(x1, "b n seg d -> b (n seg) d")
        
        for frft_layer, conv_mlp in zip(self.channel1_layers, self.channel1_mlp_blocks):
            frft_out = frft_layer(x1)
            conv_mlp_out = conv_mlp(frft_out)
            x1 = x1 + conv_mlp_out
        
        x1 = self.channel1_norm(x1)
        x1 = x1.mean(dim=1) 
        
       
        x2_wpli = self.wpli_extractor(wpli_graphs)  
        x2_gc = self.gc_extractor(gc_graphs)  
        
       
        x = torch.cat([x1, x2_wpli, x2_gc], dim=1)
        x = self.fusion_proj(x)
        x = self.fusion_norm(x)
        x = self.fusion_dropout(x)
        
      
        for block in self.conv_mlp_blocks:
            x = block(x.unsqueeze(1)).squeeze(1)
        
        
        output = self.output_layer(x)
        
        if self.config.task_name == 'classification':
            return output, None
        else:
            return output.squeeze(-1), None



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = FeatureFusionModel(config)
    
    def forward(self, x, wpli_graphs, gc_graphs, enc_self_mask=None):
        return self.model(x, wpli_graphs, gc_graphs)