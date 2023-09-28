import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

#barf
import importlib
import options
import os,sys,time
import camera
from easydict import EasyDict as edict
import torch.nn.functional as torch_F
from torch.utils import tensorboard
import util,util_vis

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)

#barf-barf
# def get_all_training_poses(opt,self_graph):
#     # get ground-truth (canonical) camera poses
#     #pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
#     pose_GT =get_all_camera_poses(opt).to(opt.device)
#     # add synthetic pose perturbation to all training data
#     pose = get_all_camera_poses(opt).to(opt.device)
#     # add learned pose correction to all training data
#     pose_refine = camera.lie.se3_to_SE3(self_graph.se3_refine.weight)
#     pose = camera.pose.compose([pose_refine,pose])
#     return pose,pose_GT

#barf-llff
def center_camera_poses(opt,poses):
    # compute average pose
    center = poses[...,3].mean(dim=0)
    v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
    v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
    v0 = v1.cross(v2)
    pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
    # apply inverse of averaged pose
    poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
    return poses

def parse_cameras_and_bounds(opt):
    fname = "data/custom3/poses_bounds.npy"
    data = torch.tensor(np.load(fname),dtype=torch.float32)
    # parse cameras (intrinsics and poses)
    cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
    poses_raw = cam_data[...,:4] # [N,3,4]
    poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0]
    raw_H,raw_W,raw_focal = cam_data[0,:,-1]
    #assert(self.raw_H==raw_H and self.raw_W==raw_W)
    raw_H = 1080
    raw_W = 1440
    # parse depth bounds
    bounds = data[:,-2:] # [N,2]
    scale = 1./(bounds.min()*0.75) # not sure how this was determined
    poses_raw[...,3] *= scale
    bounds *= scale
    # roughly center camera poses
    poses_raw = center_camera_poses(opt,poses_raw)
    return poses_raw,bounds

#barf-llff
def get_all_camera_poses(opt):
    train_data_path = "data/custom3/images"
    poses_raw, bounds = parse_cameras_and_bounds(opt)
    image_fnames = sorted(os.listdir(train_data_path))
    data_list = list(zip(image_fnames, poses_raw, bounds))

    pose_raw_all = [tup[1] for tup in data_list]
    pose_all = torch.stack([parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
    return pose_all

#barf-barf
def prealign_cameras(opt,pose,pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    opt.device = "cuda:0"
    center = torch.zeros(1,1,3,device=opt.device)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device=opt.device))
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose_aligned = camera.pose(R=R_aligned,t=t_aligned)
    return pose_aligned,sim3

#barf-llff
def parse_raw_camera(opt,pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    pose = camera.pose.invert(pose)
    pose = camera.pose.compose([pose_flip,pose])
    return pose

#barf-base-barf
# def log_scalars(opt,var,loss,metric=None,step=0,split="train",optim_pose):
#     # for key,value in loss.items():
#     #     if key=="all": continue
#     #     if opt.loss_weight[key] is not None:
#     #         self.tb.add_scalar("{0}/loss_{1}".format(split,key),value,step)
#     # if metric is not None:
#     #     for key,value in metric.items():
#     #         self.tb.add_scalar("{0}/{1}".format(split,key),value,step)
#     optimizer2 = getattr(torch.optim, opt.optim.algo)
#     self_graph.se3_refine = torch.nn.Embedding(len(train_loader), 6).to(opt_b.device)
#     optim_pose = optimizer2([dict(params=self_graph.se3_refine.parameters(), lr=opt.optim.lr_pose)])
#     if split == "train":
#         # log learning rate
#         lr = optim_pose.param_groups[0]["lr"]
#         tb.add_scalar("{0}/{1}".format(split, "lr_pose"), lr, step)
#     # compute pose error
#     if split == "train" and opt.data.dataset in ["blender", "llff"]:
#         pose, pose_GT = get_all_training_poses(opt)
#         pose_aligned, _ = prealign_cameras(opt, pose, pose_GT)
#         error = evaluate_camera_alignment(opt, pose_aligned, pose_GT)
#         tb.add_scalar("{0}/error_R".format(split), error.R.mean(), step)
#         tb.add_scalar("{0}/error_t".format(split), error.t.mean(), step)

#barf-barf
def evaluate_camera_alignment(opt,pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error

#barf-base
# def train_iteration_loss(opt,var,loader,it):
#     # before train iteration
#     #timer.it_start = time.time()
#     # train iteration
#     optimizer_b = getattr(torch.optim, opt.optim.algo)
#     optim = optimizer_b([dict(params=self_graph.parameters(), lr=opt.optim.lr)])
#     optim.zero_grad() #grad가 누적되기 때문에 0으로 초기화
#     var = graph.forward(opt,var,mode="train")
#     loss = graph.compute_loss(opt,var,mode="train")
#     loss = summarize_loss(opt,var,loss)
#     loss.all.backward() #역전
#     optim.step()
#     # after train iteration
#     if (it+1)%opt.freq.scalar==0: log_scalars(opt,var,loss,step=it+1,split="train")
#     #if (it+1)%opt.freq.vis==0: visualize(opt,var,step=it+1,split="train")
#     it += 1
#     loader.set_postfix(it=it,loss="{:.3f}".format(loss.all))
#     #self.timer.it_end = time.time()
#     #util.update_timer(opt,self.timer,self.ep,len(loader))
#     return loss

# def train_iteration_pose(opt,var,loader,it,optim_pose,self_graph,sched_pose,loss):
#     optim_pose.zero_grad()
#     if opt.optim.warmup_pose:
#         # simple linear warmup of pose learning rate
#         optim_pose.param_groups[0]["lr_orig"] = optim_pose.param_groups[0]["lr"] # cache the original learning rate
#         optim_pose.param_groups[0]["lr"] *= min(1,it/opt.optim.warmup_pose)
#     #loss = train_iteration_loss(opt,var,loader)
#     #loss.all.backward()
#     #scaler = torch.cuda.amp.GradScaler(enabled=True)
#     #scaler.scale(loss).backward()
#     #loss = edict()
#     #loss.all = (pred.contiguous()-label)**2
#     loss.backward()
#     optim_pose.step()

    # if opt.optim.warmup_pose: #x
    #     optim_pose.param_groups[0]["lr"] = optim_pose.param_groups[0]["lr_orig"] # reset learning rate
    #
    # if opt.optim.sched_pose:
    #     sched_pose.step()
    # self_graph.nerf.progress.data.fill_(it/opt.max_iter)
    # if opt.nerf.fine_sampling: #x
    #     self_graph.nerf_fine.progress.data.fill_(it/opt.max_iter)
    # #return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=35000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()
    opt.O = True
    opt.workspace = 'trial_nerf_custom'
    opt.gui = True

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1: #X
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:#X
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:#X
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed_everything(opt.seed)
    sys.argv_b = ['0', '--group=GROUP', '--model=barf', '--yaml=barf_llff', '--name=fox', '--data.scene=city_nosky',
                  '--barf_c2f=[0.1,0.5]']
    opt_cmd_b = options.parse_arguments(sys.argv_b[1:])
    opt_b = options.set(opt_cmd=opt_cmd_b)
    graph = importlib.import_module("model.{}".format(opt_b.model))

    # self_graph = graph.Graph(opt_b).to(opt_b.device)
    # self_graph.train()
    # m = graph.Model(opt_b)
    # # m.load_dataset(opt_b)
    train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
    from BARF.barf import POSE

    pose_train = POSE(opt_b, graph, train_loader)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        pose_train = pose_train
    )


    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?


    
    if opt.test: #X
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            model.cam_pose = gui.cam.pose
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:




        #barf build network part
        # sys.argv_b=['0','--group=GROUP','--model=barf','--yaml=barf_llff','--name=fox','--data.scene=city_nosky','--barf_c2f=[0.1,0.5]']
        # opt_cmd_b = options.parse_arguments(sys.argv_b[1:])
        # opt_b = options.set(opt_cmd=opt_cmd_b)
        # graph = importlib.import_module("model.{}".format(opt_b.model))
        #
        # # self_graph = graph.Graph(opt_b).to(opt_b.device)
        # # self_graph.train()
        # # m = graph.Model(opt_b)
        # # # m.load_dataset(opt_b)
        # train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        # from BARF.barf import POSE
        # pose_train = POSE(opt_b,graph,train_loader)
        # self_graph.se3_refine = torch.nn.Embedding(len(train_loader), 6).to(opt_b.device) #이산적인 데이터를 연속적으로 표현하게함
        # torch.nn.init.zeros_(self_graph.se3_refine.weight)  # tensor를 0으로 채워줌

        # model.self_graph = self_graph
        #model.se3_refine = self_graph.se3_refine
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99),
                                                   eps=1e-15)  # 최소값을 찾아가는 알고리즘

        # decay to 0.1 * init_lr at last iter step 일정한 시간마다 정해진 로직 돌리기
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        #trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50, opt_BARF = opt_b,var_BARF=,loader_BARF=)


        #barf-log_scalar
        split = "train"
        step = 0
        #optimizer2 = getattr(torch.optim, opt_b.optim.algo)
        #self_graph.se3_refine = torch.nn.Embedding(len(train_loader), 6).to(opt_b.device)
        #torch.nn.init.zeros_(self_graph.se3_refine.weight)
        # optim_pose = optimizer2([dict(params=self.pose_train.self_graph.se3_refine.parameters(), lr=opt_b.optim.lr_pose)])
        #barf pose scheduler
        # set up scheduler


        tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt_b.output_path, flush_secs=10)
        # if split=="train":
        #     # log learning rate
        #     lr = optim_pose.param_groups[0]["lr"]
        #     tb.add_scalar("{0}/{1}".format(split,"lr_pose"),lr,step)

        # pose, pose_GT = get_all_training_poses(opt_b,self_graph)
        # pose_aligned, _ = prealign_cameras(opt_b, pose, pose_GT)
        # error = evaluate_camera_alignment(opt_b, pose_aligned, pose_GT)
        # opt_b.output_path = "output/GROUP/test"
        # tb.add_scalar("{0}/error_R".format(split), error.R.mean(), step)
        # tb.add_scalar("{0}/error_t".format(split), error.t.mean(), step) #로그 기록..

        #barf
        data = importlib.import_module("data.{}".format("llff"))
        train_data = data.Dataset(opt_b, split="train", subset=None)
        train_data.prefetch_all_data(opt_b)
        train_data.all = edict(util.move_to_device(train_data.all, opt_b.device))
        var = train_data.all
        loader = tqdm.trange(opt_b.max_iter, desc="training", leave=False)
        train_dataset = NeRFDataset(opt, device=device, type='train')
        model.train_dataset = train_dataset
        #
        model.train_dataset.pred_rgb = []

        # model.train_dataset.pred_rgb = [torch.zeros((4096, 3)) for _ in range(len(model.train_dataset.images))]
        model.train_dataset.rgb = []
        # model.train_dataset.rgb = [torch.zeros((4096, 3)) for _ in range(len(model.train_dataset.images))]
       #
        model.pose_train.poses = train_dataset.poses
        # model.pose_train.network = model
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50,
                          opt_BARF=opt_b, var_BARF=var, loader_BARF=loader,
                          self_graph=pose_train.self_graph,loader = train_loader,
                          train_dataset=train_dataset,pose_train=pose_train)
        it=0
        model.trainer = trainer


        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader, device=device)
            model.cam_pose = gui.cam.pose
            gui.render()
        
        else:

            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)
            #/gui = NeRFGUI(opt, trainer, train_loader)
            #gui.render()

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video



            trainer.save_mesh(resolution=256, threshold=10)

