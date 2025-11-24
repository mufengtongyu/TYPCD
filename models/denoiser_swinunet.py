# models/denoiser_swinunet.py
"""
2025年9月18日使用ChatGPT生成
Swin-UNet style denoiser for diffusion models on 2D fields (ERA5 slices).
- Primary attempt: use timm.SwinTransformer as encoder (if timm is available).
- Fallback: a lightweight convolutional encoder (for quick testing).
- Decoder: U-Net style upsampling with FiLM modulation.
- Conditioning: c_spec and c_share injected by FiLM (first c_spec, then c_share)
- Time embedding: sinusoidal -> MLP
- Forward signature:
    forward(x, timesteps, c_spec=None, c_share=None)
  where x: [B, C, H, W], timesteps: [B] long tensor or scalar int,
  c_spec/c_share: [B, cond_dim] or None.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import timm's Swin; if not present, we'll fallback to conv encoder.
try:
    import timm
    from timm.models.swin_transformer import SwinTransformer
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

# -------------------------
# Utilities: Time embedding
# -------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        timesteps: tensor shape [B] or scalar
        returns: [B, dim]
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # pad
            emb = F.pad(emb, (0,1), value=0)
        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sin_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward(self, t):
        return self.mlp(self.sin_emb(t))

# -------------------------
# Small conv blocks & utils
# -------------------------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=True)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.act = nn.GELU()
        self.conv2 = conv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x):
        h = self.conv1(x)
        h = self.act(h)
        h = self.conv2(h)
        res = x if self.res_conv is None else self.res_conv(x)
        return self.act(h + res)

class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ResidualConvBlock(in_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # concat along channel dimension
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# -------------------------
# Conditioning projector (FiLM)
# -------------------------
class FiLMProjector(nn.Module):
    def __init__(self, cond_dim, features_channels_list):
        """
        cond_dim: condition vector dim
        features_channels_list: list of channels per decoder stage where FiLM applied
        Produces gamma/beta for each stage.
        """
        super().__init__()
        self.features_channels_list = features_channels_list
        total = sum([c*2 for c in features_channels_list])  # gamma + beta per channel
        self.net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim*2),
            nn.GELU(),
            nn.Linear(cond_dim*2, total)
        )

    def forward(self, cond):
        # cond: [B, cond_dim]
        params = self.net(cond)  # [B, total]
        splits = torch.split(params, [c*2 for c in self.features_channels_list], dim=-1)
        gammas_betas = []
        for s in splits:
            g, b = s.chunk(2, dim=-1)  # each [B, channels]
            gammas_betas.append((g, b))
        return gammas_betas  # list of (gamma, beta) for each stage

# -------------------------
# Lightweight Conv Encoder (fallback)
# -------------------------
class ConvEncoder(nn.Module):
    """A simple hierarchical conv encoder producing multi-scale features (list)."""
    def __init__(self, in_ch=1, base_ch=64, depths=[2,2,2,2]):
        super().__init__()
        chs = [base_ch * (2**i) for i in range(len(depths))]
        self.stem = ResidualConvBlock(in_ch, chs[0])
        self.downs = nn.ModuleList()
        for i, d in enumerate(depths):
            blocks = nn.Sequential(*[ResidualConvBlock(chs[i], chs[i]) for _ in range(d)])
            self.downs.append(blocks)
        # to perform downsample between scales, we'll use conv stride
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        h = self.stem(x)  # [B, C0, H, W]
        features.append(h)
        for block in self.downs:
            h = block(h)
            features.append(h)
            h = self.pool(h)
        # Return top-down list with progressively smaller spatial sizes
        return features  # list length = 1 + len(depths)

# -------------------------
# Swin-UNet Denoiser
# -------------------------
class SwinUnetDenoiser(nn.Module):
    def __init__(self,
                 in_ch=1,
                 base_dim=96,
                 tim_emb_dim=256,
                 cond_dim=256,
                 decoder_channels = [384, 192, 96, 48],
                 use_timm_swin=True,
                 swin_img_size=128,
                 swin_patch_size=4):
        """
        in_ch: input channels (ERA5 channels)
        base_dim: Swin embed dim if using timm (default 96)
        tim_emb_dim: time embedding dimension
        cond_dim: condition vector dimension (expected same for c_spec and c_share; you can concat or project externally)
        decoder_channels: channels used in decoder stages (from deep -> shallow)
        use_timm_swin: try to use timm.SwinTransformer if available (and TIMM_AVAILABLE)
        swin_img_size/pach_size: image size and patch size for Swin init
        """
        super().__init__()
        self.in_ch = in_ch
        self.cond_dim = cond_dim
        self.tim_emb = TimeEmbedding(tim_emb_dim)

        # Encoder (Swin if available & chosen)
        self.use_swin = use_timm_swin and TIMM_AVAILABLE
        if self.use_swin:
            # Create timm Swin and use its feature outputs
            # Note: timm Swin forward_features returns a single tensor for some versions.
            # We'll use feature extraction by defining stages manually if available.
            # Here we instantiate SwinTransformer but will rely on its forward_features method
            self.swin = SwinTransformer(img_size=swin_img_size,
                                        patch_size=swin_patch_size,
                                        in_chans=in_ch,
                                        embed_dim=base_dim,
                                        depths=[2,2,6,2],
                                        num_heads=[3,6,12,24],
                                        window_size=7,
                                        drop_rate=0.0,
                                        attn_drop_rate=0.0,
                                        drop_path_rate=0.1,
                                        use_checkpoint=False)
            # We'll adapt outputs from swin.features (if available) else fallback to single feature.
            # To interface with decoder, we provide projection convs from swin embed dims to decoder channels
            swin_out_chs = [base_dim*2**i for i in range(4)]  # typical Swin layer dims
            self.enc_projs = nn.ModuleList([
                nn.Conv2d(swin_out_chs[i], decoder_channels[i], kernel_size=1)
                for i in range(len(decoder_channels))
            ])
        else:
            # fallback conv encoder
            self.conv_encoder = ConvEncoder(in_ch=in_ch, base_ch=48, depths=[2,2,2])
            # provide projection convs to map conv encoder features to decoder channels
            self.enc_projs = nn.ModuleList([
                nn.Conv2d(48*(2**i), decoder_channels[i], kernel_size=1)
                for i in range(len(decoder_channels))
            ])

        # Middle (bottleneck)
        self.bottleneck = ResidualConvBlock(decoder_channels[0], decoder_channels[0])

        # Decoder: upsample blocks with skip connections
        # For skip, we expect enc_projs in same order as decoder_channels
        self.decoder_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        # features channels used for FiLM modulation (we apply FiLM at each decoder block input)
        film_features = []
        for i in range(len(decoder_channels)-1):
            in_ch_dec = decoder_channels[i] + decoder_channels[i+1]  # concat with skip
            out_ch = decoder_channels[i+1]
            self.up_blocks.append(UpSampleBlock(in_ch=decoder_channels[i], out_ch=out_ch))
            # after upsample + concat, apply a residual block to refine
            self.decoder_blocks.append(ResidualConvBlock(out_ch + out_ch, out_ch))  # conservative
            film_features.append(out_ch)
        # final up to input resolution
        self.final_block = ResidualConvBlock(decoder_channels[-1] + decoder_channels[-1], decoder_channels[-1])
        film_features.append(decoder_channels[-1])

        # Projection to image channels (predict noise)
        self.out_conv = nn.Sequential(
            conv3x3(decoder_channels[-1], decoder_channels[-1]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[-1], in_ch, kernel_size=1)
        )

        # FiLM projectors: one for c_spec and one for c_share (we'll combine sequentially)
        # Each produces gammas & betas for each film stage (in decoder order)
        self.film_spec = FiLMProjector(cond_dim, film_features)
        self.film_share = FiLMProjector(cond_dim, film_features)

        # Optional small projector for time embedding to match feature dims (applied as addition)
        self.time_proj = nn.ModuleList([nn.Sequential(nn.Linear(tim_emb_dim, ch), nn.GELU()) for ch in film_features])

        # Initialization helper
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_features(self, x):
        """
        Returns a list of encoder feature maps ordered deep->shallow to match decoder_channels.
        Each feature projected to decoder channel size via enc_projs.
        """
        if self.use_swin:
            # timm swin's forward_features might return last feature only depending on version.
            # We'll attempt to extract features by calling swin.forward_features and then slicing.
            # Note: this part may need slight adaptation depending on timm version.
            try:
                feats = self.swin.forward_features(x)  # might be single tensor
                # If single tensor, we derive multi-scale features by simple pooling splits
                if isinstance(feats, torch.Tensor):
                    # create a simple pyramid by pooling
                    f0 = feats  # deepest
                    f1 = F.adaptive_avg_pool2d(f0, (f0.shape[2]//2, f0.shape[3]//2))
                    f2 = F.adaptive_avg_pool2d(f1, (f1.shape[2]//2, f1.shape[3]//2))
                    f3 = F.adaptive_avg_pool2d(f2, (f2.shape[2]//2, f2.shape[3]//2))
                    feat_list = [f0, f1, f2, f3]
                else:
                    # If swin returns list-like
                    feat_list = feats
            except Exception:
                # worst-case fallback: single pass through swin, then pooling
                feats = self.swin.forward(x)
                feat_list = [feats]
            # project to decoder channel dims
            proj_feats = []
            # choose up to number of enc_projs (decoder stages)
            for i, proj in enumerate(self.enc_projs):
                # guard index
                idx = min(i, len(feat_list)-1)
                f = feat_list[idx]
                # if channel mismatch, apply 1x1
                proj_feats.append(proj(f))
            # ensure order deep->shallow (decoder expects [deep,...,shallow])
            return proj_feats
        else:
            conv_feats = self.conv_encoder(x)  # list shallow -> deep (depending on implementation)
            # our ConvEncoder returns [stem, block1, block2, ...] so we pick last ones
            proj_feats = []
            for i, proj in enumerate(self.enc_projs):
                idx = min(i, len(conv_feats)-1)
                f = conv_feats[idx]
                proj_feats.append(proj(f))
            return proj_feats

    def apply_film(self, feat, gamma, beta):
        """
        feat: [B, C, H, W], gamma/beta: [B, C]
        apply FiLM: feat * (1 + gamma.view) + beta.view
        """
        B, C, H, W = feat.shape
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return feat * (1.0 + gamma) + beta

    def forward(self, x, timesteps, c_spec=None, c_share=None):
        """
        x: [B, C, H, W]
        timesteps: [B] long tensor or scalar
        c_spec, c_share: [B, cond_dim] or None
        """
        B = x.shape[0]
        t_emb = self.tim_emb(timesteps)  # [B, tim_emb_dim]

        # encoder
        enc_feats = self.encode_features(x)  # list deep->shallow (length = num stages)
        # ensure we have at least as many features as decoder stages
        # We'll treat enc_feats[0] as deepest, enc_feats[-1] as shallowest
        # pass through bottleneck
        deep_feat = enc_feats[0]
        h = self.bottleneck(deep_feat)

        # prepare FiLM params
        device = x.device
        if c_spec is None:
            c_spec = torch.zeros(B, self.cond_dim, device=device)
        if c_share is None:
            c_share = torch.zeros(B, self.cond_dim, device=device)

        film_spec_params = self.film_spec(c_spec)  # list per stage
        film_share_params = self.film_share(c_share)

        # decoder loop: we assume enc_feats length matches film lists
        # We'll iterate from deep index 0 -> to shallow
        cur = h
        num_stages = len(self.up_blocks)
        for i in range(num_stages):
            # upsample current (from stage i deep -> i+1)
            # enc_feats[i+1] is the corresponding skip (slightly shallower)
            skip = enc_feats[i+1] if (i+1) < len(enc_feats) else None
            # up_blocks expect input channels match decoder_channels[i], but we used
            # projections to ensure shapes are compatible
            cur = self.up_blocks[i](cur, skip)
            # after upsample we can apply FiLM modulation for this stage
            # film params are ordered corresponding to film_features list in __init__
            gamma_spec, beta_spec = film_spec_params[i]
            gamma_share, beta_share = film_share_params[i]
            # optionally add time embedding (projected) to feature via broadcast add
            tproj = self.time_proj[i](t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            cur = cur + tproj
            # apply spec then share
            cur = self.apply_film(cur, gamma_spec, beta_spec)
            cur = self.apply_film(cur, gamma_share, beta_share)
            # refine with residual block if present
            if i < len(self.decoder_blocks):
                # prepare skip concat for this refinement block (we duplicate skip channel to match expected in block)
                if skip is not None:
                    # make shapes compatible by simple concat (skip may already have proper channel)
                    cur = torch.cat([cur, skip], dim=1)
                cur = self.decoder_blocks[i](cur)

        # final refinement: concat with shallowest skip if available
        shallow_skip = enc_feats[-1] if len(enc_feats) > 1 else None
        if shallow_skip is not None:
            # if mismatch in spatial size, upsample shallow_skip
            if shallow_skip.shape[2:] != cur.shape[2:]:
                shallow_skip = F.interpolate(shallow_skip, size=cur.shape[2:], mode='bilinear', align_corners=False)
            cur = torch.cat([cur, shallow_skip], dim=1)
        cur = self.final_block(cur)

        out = self.out_conv(cur)
        # predict noise residual -> same shape as x
        return out

# -------------------------
# Convenience factory
# -------------------------
def build_swin_unet_denoiser(cfg=None):
    """
    cfg: optional dict with keys to control architecture.
    Example:
      cfg = {
         'in_ch':1, 'base_dim':96, 'tim_emb_dim':256, 'cond_dim':256,
         'decoder_channels':[384,192,96,48], 'use_timm_swin':True,
         'swin_img_size':128, 'swin_patch_size':4
      }
    """
    if cfg is None:
        cfg = {}
    return SwinUnetDenoiser(
        in_ch=cfg.get('in_ch', 1),
        base_dim=cfg.get('base_dim', 96),
        tim_emb_dim=cfg.get('tim_emb_dim', 256),
        cond_dim=cfg.get('cond_dim', 256),
        decoder_channels=cfg.get('decoder_channels', [384, 192, 96, 48]),
        use_timm_swin=cfg.get('use_timm_swin', True),
        swin_img_size=cfg.get('swin_img_size', 128),
        swin_patch_size=cfg.get('swin_patch_size', 4)
    )
