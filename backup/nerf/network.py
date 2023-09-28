import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 # se3_refine=None,
                 # graph=None,
                 pose_train=None,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.pose_train = pose_train
        torch.nn.init.zeros_(self.pose_train.self_graph.se3_refine.weight)
        # self.train_dataset = NeRFDataset(opt, device=device, type='train')

        #barf
        # if cuda.ray:
        #     step_counter = torch.zeros(16, 2, dtype=torch.int32) # 16 is hardcoded for averaging...
        #     self.register_buffer('step_counter', step_counter)
        #     step_counter.zero_()
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
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


    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        #barf

        # if self.mean_count <= 0: x = self.encoder(x, self, bound=self.bound)
        # else:
        #     if (self.training):
        #
        #         import random
        #         from nerf.utils import get_rays
        #         # from BARF.barf import pose_train
        #         #1,4,4 = poses
        #         # 포즈 업데이트하기
        #         # xyzs, dirs, deltas, rays = self.pose_train(self.trainer,self)
        #         #index = self.trainer.index
        #         # error_map = None if self.train_dataset.error_map is None else self.train_dataset.error_map[index]
        #         # rays = get_rays(poses, self.train_dataset.intrinsics, self.train_dataset.H,
        #         #                 self.train_dataset.W,
        #         #                 self.train_dataset.num_rays, error_map, self.train_dataset.opt.patch_size)
        #         # rays_o = rays['rays_o']  # [B, N, 3] 원점
        #         # rays_d = rays['rays_d']  # [B, N, 3] 방향
        #         #
        #         # rays_o = rays_o.contiguous().view(-1, 3)  #: ray 시작점 위치
        #         # rays_d = rays_d.contiguous().view(-1, 3)  # ray 방향
        #         # import raymarching
        #         # nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
        #         #                                              self.aabb_train if self.training else self.aabb_infer,
        #         #                                              self.min_near)
        #         # counter = self.step_counter[self.local_step % 16]
        #         #
        #         # counter.zero_()  # set to 0
        #         # xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
        #         #                                                         self.cascade, self.grid_size, nears, fars,
        #         #                                                         counter, self.mean_count, True, 128, False,
        #         #                                                         self.trainer.dt_gamma)
        #         x = self.encoder(xyzs, self, bound=self.bound)  # gridencoder
        #         d = dirs
        # # sigma
        #     else:
        #     # x = self.encoder(x, bound=self.bound)
        #         x = self.encoder(x, self, bound=self.bound)
        #      # x = self.encoder(x,self.train_dataset, self,self.trainer,bound=self.bound) #gridencoder

        x = self.encoder(x, bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0]) #trunc forward
        geo_feat = h[..., 1:]

        # color

        #pose
        # poses = self.pose_train(self)
        # d.requires_grad_(True)
        d = self.encoder_dir(d) #shencoder forward 구면조화함수 계<
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        # if (self.training):
        #     x = self.encoder(x, self.train_dataset,self,self.trainer,poses=self.pose_train.poses, bound=self.bound)
        # else : x = self.encoder.forward2(x,self, bound=self.bound)

        x = self.encoder(x,bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

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
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},

        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
