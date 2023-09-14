import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

from transformers import CLIPImageProcessor, CLIPVisionModel

class SemanticNetwork(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32') -> None:
        super().__init__()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.eval()

    @torch.inference_mode()
    def forward(self, images):
        # images [B, 3, H, W]
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.pooler_output # [B, 768]


class MultiViewEncoder(nn.Module):
    def __init__(self, vision_final_dim=500, final_dim=256) -> None:
        super().__init__()
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        self.vision_preprocess = weights.transforms(antialias=True)
        self.vision_model = efficientnet_b2(weights=weights)
        self.vision_model.classifier = nn.Identity()
        
        self.vision_mlp = nn.Linear(1408, vision_final_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(vision_final_dim + 12, final_dim), # TODO: modify this, add dropout, activation, etc.
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(final_dim, final_dim),
        )
        self.tanh = nn.Tanh()

    def forward(self, multi_images, multi_poses):
        # multi_images [B, 3, H, W] 
        # multi_poses [B, 3, 4]
        # where B is always the number of multi-views
        B = multi_images.shape[0]
        multi_poses = multi_poses.view(B, -1) # [B, 12]

        preprocessed_img = self.vision_preprocess(multi_images)
        img_features = self.vision_model(preprocessed_img)
        img_features = self.vision_mlp(img_features) # [B, vision_final_dim]

        features = torch.cat([img_features, multi_poses], dim=1) # [B, vision_final_dim + 12]
        features = self.mlp1(features)
        
        features = features.mean(dim=0, keepdim=True) # [1, final_dim]
        features = self.mlp2(features)
        features = self.tanh(features)

        return features # [1, final_dim]


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency", # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5, # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 latent_vector_dim=256,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)
        self.latent_vector_dim = latent_vector_dim
        
        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                # in_dim = self.in_dim_deform + self.in_dim_time # grid dim + time
                # in_dim = self.in_dim_deform + self.latent_vector_dim # grid dim + latent vector dim
                in_dim = self.in_dim_deform + self.latent_vector_dim + self.in_dim_time # grid dim + latent vector dim + time
            else:
                in_dim = hidden_dim_deform
            
            if l == num_layers_deform - 1:
                out_dim = 3 # deformation for xyz
            else:
                out_dim = hidden_dim_deform
            
            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)


        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                # in_dim = self.in_dim + self.in_dim_time + self.in_dim_deform # concat everything
                # in_dim = self.in_dim + self.latent_vector_dim + self.in_dim_deform # concat everything
                in_dim = self.in_dim + self.latent_vector_dim + self.in_dim_deform + self.in_dim_time # concat everything
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

        # multi-view encoder
        self.multiview_enc = MultiViewEncoder(final_dim=latent_vector_dim)

    def forward(self, x, d, t, multi_images, multi_poses):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]
        # multi_images [B, 3, H, W] 
        # multi_poses [B, 3, 4]
        # print(x.shape)
        # print(d.shape)
        # print(t.shape)
        # print(multi_images.shape)
        # print(multi_poses.shape)
        latent_vector = self.multiview_enc(multi_images, multi_poses) # [1, final_dim]
        latent_vector = latent_vector.repeat(x.shape[0], 1) # [1, final_dim] --> [N, final_dim]

        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        # deform = torch.cat([enc_ori_x, enc_t], dim=1) # [N, C + C']
        # deform = torch.cat([enc_ori_x, latent_vector], dim=1) # [N, C + C']
        deform = torch.cat([enc_ori_x, enc_t, latent_vector], dim=1) # [N, C + C' + latent_vector_dim]
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        x = x + deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        # h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        # h = torch.cat([x, enc_ori_x, latent_vector], dim=1)
        h = torch.cat([x, enc_ori_x, enc_t, latent_vector], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)
        return sigma, rgbs, deform

    def latent_vector(self, multi_images, multi_poses):
        # multi_images [B, 3, H, W] 
        # multi_poses [B, 3, 4]
        return self.multiview_enc(multi_images, multi_poses) # [1, final_dim]

    def density(self, x, t, multi_images, multi_poses):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # multi_images [B, 3, H, W] 
        # multi_poses [B, 3, 4]
        results = {}
        latent_vector = self.multiview_enc(multi_images, multi_poses) # [1, final_dim]
        latent_vector = latent_vector.repeat(x.shape[0], 1) # [1, final_dim] --> [N, final_dim]
        # deformation
        enc_ori_x = self.encoder_deform(x, bound=self.bound) # [N, C]
        enc_t = self.encoder_time(t) # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1) # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t, latent_vector], dim=1) # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)
        
        x = x + deform
        results['deform'] = deform
        
        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t, latent_vector], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.deform_net.parameters(), 'lr': lr_net},
            {'params': self.multiview_enc.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})
        
        return params

if __name__ == '__main__':
    nerf = NeRFNetwork()
    nerf.to(0)
    N = 100
    x = torch.randn(N, 3).to(0)
    d = torch.randn(N, 3).to(0)
    t = torch.randn(1, 1).to(0)
    multi_images = torch.randn(4, 3, 800, 800).to(0)
    multi_poses = torch.randn(4, 3, 4).to(0)

    res = nerf(x, d, t, multi_images, multi_poses)