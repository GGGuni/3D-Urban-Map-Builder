import torch
import camera
from torch.autograd import Function
import random
from easydict import EasyDict as edict
from packaging import version as pver

import numpy as np

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import FunctionMeta
# try:
#     # import _ffmlp as _backend
#     import _pose_train as _backend
# except ImportError:
#     pass
#     from .backend import _backend
# try:
#     import _pose_train as _backend
# except ImportError:
#     from .backend import _backend

#
# def custom_meshgrid(*args):
#     # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
#     if pver.parse(torch.__version__) < pver.parse('1.10'):
#         return torch.meshgrid(*args)
#     else:
#         return torch.meshgrid(*args, indexing='ij')
#
# def srgb_to_linear(x):
#     return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

class _pose_train(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx,se3,poses,device,trainer,network,index=None):#,  # poses : new pose , w : se3_refine
        # 수정된 new pose로 이미지 렌더링 해서 rgb loss를 return
        # inputs = input.contiguous()

        # print("barf forward")
        # if trainer.training:


        if network.training :
            import camera
            ## 이전에 업데이트 한 se3_refine weight으로 새 pose 구하기
            poses = poses[:, :-1]  # 1,3,4

            # tensor_idx = torch.arange(40)
            # var = se3.weight[tensor_idx]
            poses_refine = camera.lie.se3_to_SE3(se3)
            poses_new = camera.pose.compose([poses_refine, poses])
            a = poses_new[index]
            new_row = torch.tensor([[[0, 0, 0, 1]]]).to(device)  # 1,3,4
            # new_row = change_matrix(poses, scale=0.33, offset=[0, 0, 0])
            poses = torch.cat([a, new_row], dim=1)

            trainer.train_dataset.poses[index[0]] = poses
            # wu = camera.lie.SE3_to_se3()
            #-------------
            error_map = None if trainer.train_dataset.error_map is None else trainer.train_dataset.error_map[index]
            from nerf.utils import get_rays
            rays = get_rays(poses, trainer.train_dataset.intrinsics, trainer.train_dataset.H,
                            trainer.train_dataset.W,
                            trainer.train_dataset.num_rays, error_map, trainer.train_dataset.opt.patch_size,
                            trainer.i)
            rays_o = rays['rays_o']  # [B, N, 3] 원점
            rays_d = rays['rays_d']  # [B, N, 3] 방향


            rays_o = rays_o.contiguous().view(-1, 3)  #: ray 시작점 위치
            rays_d = rays_d.contiguous().view(-1, 3)  # ray 방향
            import raymarching
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                         network.aabb_train if network.training else network.aabb_infer,
                                                         network.min_near)

            counter = network.step_counter[network.local_step % 16]

            counter.zero_()  # set to 0
            # if trainer.training:
            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, network.bound, network.density_bitfield,
                                                                    network.cascade, network.grid_size, nears, fars,
                                                                    counter, network.mean_count, True, 128, False,
                                                                    network.trainer.dt_gamma,1024)


            ctx.save_for_backward(se3, poses, xyzs, dirs,deltas,rays,torch.tensor(index[0]),rays_o,rays_d)

            return xyzs, dirs, deltas, rays,poses,nears,fars,se3
        else :
            import camera
            # tensor_idx = torch.arange(40)
            # var = se3.weight[tensor_idx]
            poses = poses[:-1, :]
            poses_refine = camera.lie.se3_to_SE3(se3)
            poses_new = camera.pose.compose([poses_refine, poses])
            new_row = torch.tensor([[[0, 0, 0, 1]]]).to(device)  # 1,3,4
            a = poses_new[0]
            # new_row = change_matrix(poses, scale=0.33, offset=[0, 0, 0])
            poses = torch.cat([a.unsqueeze(0), new_row], dim=1)

            error_map = None if trainer.train_dataset.error_map is None else trainer.train_dataset.error_map[index]
            from nerf.utils import get_rays
            rays = get_rays(poses, trainer.train_dataset.intrinsics, trainer.train_dataset.H,
                            trainer.train_dataset.W,
                            trainer.train_dataset.num_rays, error_map, trainer.train_dataset.opt.patch_size)
            rays_o = rays['rays_o']  # [B, N, 3] 원점
            rays_d = rays['rays_d']  # [B, N, 3] 방향

            rays_o = rays_o.contiguous().view(-1, 3)  #: ray 시작점 위치
            rays_d = rays_d.contiguous().view(-1, 3)  # ray 방향
            import raymarching
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                         network.aabb_train if network.training else network.aabb_infer,
                                                         network.min_near)
            counter = network.step_counter[network.local_step % 16]

            counter.zero_()  # set to 0
            # if trainer.training:
            N = rays_o.shape[0]
            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device) # [N]
            rays_t = nears.clone() # [N]
            n_alive = rays_alive.shape[0]
            n_step = max(min(N // n_alive, 8), 1)
            xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                              network.bound, network.density_bitfield, network.cascade,
                                                              network.grid_size, nears, fars, 128,  False, network.trainer.dt_gamma, 1024)
            return xyzs, dirs, deltas, nears, fars

    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, xyzs, dirs, deltas, rays,poses,nears,fars,se3):

        se3, pose, xyzs, dirs,deltas,rays,index,rays_o,rays_d = ctx.saved_tensors
        grad_se3 = se3.clone()
        goutput = grad_se3.clone()
        # torch.save(grad_se3, 'gradient.pth')
        # print("barf backward")
        # print("xyzs",xyzs)
        # print("dirs",dirs)
        # print("deltas",deltas)
        # print("se3",se3)
        # print("nears", nears)
        # print("fars", fars)
        # print("rays", rays)
        # print("se3: ", g)
        # print("backward xyz: ", poses)
        #print("se3: ", se3)
        grad_se3[index][0] = 0  # rx
        grad_se3[index][1] = 0
        grad_se3[index][2] = 0
        grad_se3[index][3] = 0  # tx
        grad_se3[index][4] = 0  # ty
        grad_se3[index][5] = 0
        import sympy as sp
        import camera

        j = 0
        directions = torch.load('directions.pth')
        loss = torch.load('loss.pth')
        composite_rgb_grad = torch.load('gradient_rgb.pth').unsqueeze(-1)
        grad_dir = torch.load('grad_dir.pth').unsqueeze(1)
        for i in range(20):

            # trunc_dir = torch.load('grad_trunc.pth').unsqueeze(1)
            # grad_input = torch.load('grad_inputs.pth').unsqueeze(1)
            cur_ind = rays[i][0]
            if cur_ind > len(xyzs): pass
            # tx = (xyzs[i][0] - rays_o[cur_ind][0]) / rays_d[cur_ind][0]
            # ty = (xyzs[i][1] - rays_o[cur_ind][1]) / rays_d[cur_ind][1]
            else:
                tz = (xyzs[i][2] - rays_o[cur_ind][2]) / rays_d[cur_ind][2]

                # if tx==ty and ty==tz :
                    # # 편미분 계산
                dr1 = directions[0][cur_ind][0]
                dr2 = directions[0][cur_ind][0]
                dr3 = directions[0][cur_ind][0]
                dr4 = directions[0][cur_ind][1]
                dr5 = directions[0][cur_ind][1]
                dr6 = directions[0][cur_ind][1]
                dr7 = directions[0][cur_ind][2]
                dr8 = directions[0][cur_ind][2]
                dr9 = directions[0][cur_ind][2]

                # M = torch.tensor([[dr1, dr2, dr3,1], [dr4, dr5, dr6,1], [dr7, dr8, dr9,1]]).expand(len(composite_rgb_grad), -1, -1)
                M = torch.tensor([[dr1, -dr3,-dr2, 1], [dr4, -dr6,-dr5, 1], [dr7, -dr9,-dr8, 1]]).expand(
                    len(composite_rgb_grad), -1, -1)
                se3 = torch.matmul(torch.matmul(composite_rgb_grad, grad_dir), M.cuda())

                # se3 =torch.matmul(torch.matmul(torch.matmul(composite_rgb_grad, trunc_dir),grad_input), M.cuda())
                # _backend.pose_train_backward(composite_rgb_grad, grad_dir, M, se3)
                # A = torch.tensor([[dr1,dr2,dr3,1],[dr4,dr5,dr6,1],[dr7,dr8,dr9,1]]).cuda()
                se3 = camera.lie.SE3_to_se3(se3)
                # print("se3", se3)

                if torch.any(torch.isnan(se3)) : pass
                else:
                    j += 1
                    grad_se3[index] = loss * torch.mean(se3, dim=0, keepdim=True) / j
                    # grad_se3[index][0] += se3[0]  # rx
                    # grad_se3[index][1] += se3[1]
                    # grad_se3[index][2] += se3[2]
                    # grad_se3[index][3] += se3[3]  # rx
                    # grad_se3[index][4] += se3[4]
                    # grad_se3[index][5] += se3[5]
                # print("se3_0: ", grad_se3)


        # grad_se3[index] = torch.zeros(1,6)
        # grad_se3[index][0] /=  1 # rx
        # grad_se3[index][1] /=  1
        # grad_se3[index][2] /=  1
        # grad_se3[index][3] /=  1  # tx
        # grad_se3[index][4] /=  1  # ty
        # grad_se3[index][5] /=  1
        # print("se3_1: ", grad_se3)
        # print("se3_2: ", grad_se3)
        # grad_se3 =

        return grad_se3,None,None,None,None,None

pose_train_please = _pose_train.apply

class POSE(torch.nn.Module):

    def __init__(self,opt_b,graph,train_loader):
        super().__init__()

        self.self_graph=graph.Graph(opt_b).to(opt_b.device)

        self.self_graph.se3_refine = torch.nn.Embedding(len(train_loader), 6).to(opt_b.device)
        # nn.Parameter(self.self_graph.se3_refine)
        torch.nn.init.zeros_(self.self_graph.se3_refine.weight)
        #self.poses = self.poses.to(opt_b.device)
    def forward(self,trainer,network):
        # trainer.train_dataset.poses.requires_grad_(True)
        self.self_graph.se3_refine.weight.requires_grad_(True)
        #poses = pose_train_please(trainer.model.pose_train.self_graph.se3_refine,trainer.train_dataset.poses,trainer.train_dataset.device)
        # if trainer.training:
        if network.training :
            xyzs, dirs, deltas, rays,poses,nears,fars,_ = pose_train_please(self.self_graph.se3_refine.weight, trainer.train_dataset.poses[trainer.index],
                                          trainer.train_dataset.device,trainer,network,trainer.index)
            return xyzs, dirs, deltas, rays, nears, fars
        else :
            xyzs, dirs, deltas,nears,fars = pose_train_please(self.self_graph.se3_refine.weight,torch.tensor(network.cam_pose).cuda(),
                                          trainer.train_dataset.device,trainer,network)
        # else :
        #     xyzs, dirs, deltas, rays,poses,nears,fars,_ = pose_train_please(self.self_graph.se3_refine.weight, trainer.train_dataset.poses[trainer.index],
        #                               trainer,network,trainer.train_dataset.device)
        #poses = pose_train_please(self.self_graph.se3_refine, trainer.train_dataset.poses,
         #                         trainer.train_dataset.device,trainer.index,trainer,self)
        # self.poses = poses
        # a = trainer.index
        # trainer.train_dataset.poses[a] = poses
            return xyzs, dirs, deltas,nears,fars



