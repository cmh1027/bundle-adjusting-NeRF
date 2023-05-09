import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict

import lpips
from external.pohsun_ssim import pytorch_ssim

import util,util_vis
from util import log,debug
from . import base
import camera


# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)
        self.lpips_loss = lpips.LPIPS(net="alex").to(opt.device)

    def load_dataset(self,opt,eval_split="val"):
        super().load_dataset(opt,eval_split=eval_split)
        # prefetch all training data
        if opt.data.dataset != 'phototourism':
            self.train_data.prefetch_all_data(opt)
            self.train_data.all = edict(util.move_to_device(self.train_data.all,opt.device,exclude=['feat_gt']))

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...")
        optimizer = getattr(torch.optim,opt.optim.algo)
        params = list(self.graph.nerf.parameters())
        if opt.transient.encode:
            params += [self.graph.embedding_t.weight]
        self.optim = optimizer([dict(params=params,lr=opt.optim.lr)])
        if opt.nerf.fine_sampling:
            assert(opt.transient.encode is False), "t_embedding for fine sampling is not implemented!"
            self.optim.add_param_group(dict(params=self.graph.nerf_fine.parameters(),lr=opt.optim.lr))
        # set up scheduler
        if opt.optim.sched:
            scheduler = getattr(torch.optim.lr_scheduler,opt.optim.sched.type)
            if opt.optim.lr_end:
                assert(opt.optim.sched.type=="ExponentialLR")
                if opt.data.dataset == "phototourism":
                    opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./(opt.max_iter*len(self.train_loader)))
                else:
                    opt.optim.sched.gamma = (opt.optim.lr_end/opt.optim.lr)**(1./opt.max_iter)
            kwargs = { k:v for k,v in opt.optim.sched.items() if k!="type" }
            self.sched = scheduler(self.optim,**kwargs)

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.graph.train()
        self.ep = 0 # dummy for timer
        if opt.data.dataset != 'phototourism':
            if self.iter_start==0: self.validate(opt,0)
            loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
            for self.it in loader:
                if self.it<self.iter_start: continue
                # set var to all available images
                var = self.train_data.all
                self.train_iteration(opt,var,loader)
                if opt.optim.sched: self.sched.step()
                if self.it%opt.freq.val==0: self.validate(opt,self.it)
                if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=None,it=self.it)
                if self.it == opt.early_stop:
                    print(f"Early stop at {self.it} iter")
                    break
        else:
            max_iter = int(600000 * 2048 / (len(self.train_loader) * opt.nerf.rand_rays))
            for epoch in tqdm.trange(max_iter,desc="training",leave=False):
                loader = tqdm.tqdm(self.train_loader,desc="training epoch {}".format(epoch+1),leave=False)
                for i_pic, batch in enumerate(loader):
                    self.it = epoch * len(loader) + i_pic
                    # train iteration
                    var = edict(batch)
                    var = util.move_to_device(var,opt.device)
                    self.train_iteration(opt,var,loader)
                    if opt.optim.sched: self.sched.step()
                    if self.it%opt.freq.val==0: self.validate(opt,self.it)
                    if self.it%opt.freq.ckpt==0: self.save_checkpoint(opt,ep=epoch,it=self.it)
        # after training
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def log_scalars(self,opt,var,loss,metric=None,step=0,split="train",mse=None):
        super().log_scalars(opt,var,loss,metric=metric,step=step,split=split)
        # log learning rate
        if split=="train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
            if opt.nerf.fine_sampling:
                lr = self.optim.param_groups[1]["lr"]
                self.tb.add_scalar("{0}/{1}".format(split,"lr_fine"),lr,step)
        # compute PSNR
        assert(mse is not None)
        if not opt.feature.encode:
            psnr = -10*mse.coarse.log10()
            self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
            if opt.nerf.fine_sampling:
                psnr = -10*mse.fine.log10()
                self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)
        else:
            coef = self.graph.nerf.get_refine_coef(opt)
            if coef > 0:
                psnr = -10*mse.coarse.log10()
                self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)
                if opt.nerf.fine_sampling:
                    psnr = -10*mse.fine.log10()
                    self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine"),psnr,step)
            if coef < 1:
                psnr = -10*mse.coarse_feat.log10()
                self.tb.add_scalar("{0}/{1}".format(split,"PSNR_feat"),psnr,step)
                if opt.nerf.fine_sampling:
                    psnr = -10*mse.fine_feat.log10()
                    self.tb.add_scalar("{0}/{1}".format(split,"PSNR_fine_feat"),psnr,step)
        
    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train",eps=1e-10):
        if 'W' in var and 'H' in var: W, H = int(var.W), int(var.H)
        else: W, H = opt.W, opt.H
        if opt.tb:
            util_vis.tb_image(opt,self.tb,step,split,"image",var.image)
            if split!="train":
                invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
                invdepth_static = (1-var.static_depth)/var.opacity if opt.camera.ndc else 1/(var.static_depth/var.opacity+eps)
                rgb_map = var.rgb.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
                static_rgb_map = var.static_rgb.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
                invdepth_map_static = invdepth_static.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
                util_vis.tb_image(opt,self.tb,step,split,"rgb",rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"static_rgb",static_rgb_map)
                util_vis.tb_image(opt,self.tb,step,split,"invdepth",invdepth_map)
                util_vis.tb_image(opt,self.tb,step,split,"static_invdepth",invdepth_map_static)
                if opt.nerf.fine_sampling:
                    invdepth = (1-var.depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.depth_fine/var.opacity_fine+eps)
                    invdepth_static = (1-var.static_depth_fine)/var.opacity_fine if opt.camera.ndc else 1/(var.static_depth_fine/var.opacity_fine+eps)
                    rgb_map = var.rgb_fine.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
                    static_rgb_map = var.static_rgb_fine.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
                    invdepth_map = invdepth.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
                    invdepth_map_static = invdepth_static.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
                    util_vis.tb_image(opt,self.tb,step,split,"rgb_fine",rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"static_rgb_fine",static_rgb_map)
                    util_vis.tb_image(opt,self.tb,step,split,"invdepth_fine",invdepth_map)
                    util_vis.tb_image(opt,self.tb,step,split,"static_invdepth_fine",invdepth_map_static)

    @torch.no_grad()
    def get_all_training_poses(self,opt):
        # get ground-truth (canonical) camera poses
        pose_GT = self.train_data.get_all_camera_poses(opt).to(opt.device)
        return None,pose_GT

    @torch.no_grad()
    def evaluate_full(self,opt,eps=1e-10):
        self.graph.eval()
        loader = tqdm.tqdm(self.test_loader,desc="evaluating",leave=False)
        res = []
        test_path = "{}/test_view".format(opt.output_path)
        os.makedirs(test_path,exist_ok=True)
        for i,batch in enumerate(loader):
            var = edict(batch)
            if 'W' in var and 'H' in var: W, H = int(var.W), int(var.H)
            else: W, H = opt.W, opt.H
            var = util.move_to_device(var,opt.device)
            if opt.model=="barf" and opt.optim.test_photo:
                # run test-time optimization to factorize imperfection in optimized poses from view synthesis evaluation
                var = self.evaluate_test_time_photometric_optim(opt,var)
            var = self.graph.forward(opt,var,mode="eval")
            # evaluate view synthesis
            invdepth = (1-var.depth)/var.opacity if opt.camera.ndc else 1/(var.depth/var.opacity+eps)
            rgb_map = var.static_rgb.view(-1,H,W,3).permute(0,3,1,2) # [B,3,H,W]
            invdepth_map = invdepth.view(-1,H,W,1).permute(0,3,1,2) # [B,1,H,W]
            psnr = -10*self.graph.MSE_loss(rgb_map,var.image).log10().item()
            ssim = pytorch_ssim.ssim(rgb_map,var.image).item()
            lpips = self.lpips_loss(rgb_map*2-1,var.image*2-1).item()
            res.append(edict(psnr=psnr,ssim=ssim,lpips=lpips))
            # dump novel views
            # torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(test_path,i))
            # torchvision_F.to_pil_image(var.image.cpu()[0]).save("{}/rgb_GT_{}.png".format(test_path,i))
            # torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(test_path,i))
        # show results in terminal
        print("--------------------------")
        print("PSNR:  {:8.2f}".format(np.mean([r.psnr for r in res])))
        print("SSIM:  {:8.2f}".format(np.mean([r.ssim for r in res])))
        print("LPIPS: {:8.2f}".format(np.mean([r.lpips for r in res])))
        print("--------------------------")
        # dump numbers to file
        quant_fname = "{}/quant.txt".format(opt.output_path)
        with open(quant_fname,"w") as file:
            for i,r in enumerate(res):
                file.write("{} {} {} {}\n".format(i,r.psnr,r.ssim,r.lpips))

    @torch.no_grad()
    def generate_videos_synthesis(self,opt,eps=1e-10):
        self.graph.eval()
        if opt.data.dataset=="blender":
            test_path = "{}/test_view".format(opt.output_path)
            # assume the test view synthesis are already generated
            print("writing videos...")
            rgb_vid_fname = "{}/test_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/test_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(test_path,depth_vid_fname))
        else:
            pose_pred,pose_GT = self.get_all_training_poses(opt)
            poses = pose_pred if opt.model=="barf" else pose_GT
            if opt.model=="barf" and opt.data.dataset=="llff":
                _,sim3 = self.prealign_cameras(opt,pose_pred,pose_GT)
                scale = sim3.s1/sim3.s0
            else: scale = 1
            # rotate novel views around the "center" camera of all poses
            idx_center = (poses-poses.mean(dim=0,keepdim=True))[...,3].norm(dim=-1).argmin()
            pose_novel = camera.get_novel_view_poses(opt,poses[idx_center],N=60,scale=scale).to(opt.device)
            # render the novel views
            novel_path = "{}/novel_view".format(opt.output_path)
            os.makedirs(novel_path,exist_ok=True)
            pose_novel_tqdm = tqdm.tqdm(pose_novel,desc="rendering novel views",leave=False)
            intr = edict(next(iter(self.test_loader))).intr[:1].to(opt.device) # grab intrinsics
            for i,pose in enumerate(pose_novel_tqdm):
                ret = self.graph.render_by_slices(opt,pose[None],intr=intr) if opt.nerf.rand_rays else \
                      self.graph.render(opt,pose[None],intr=intr)
                invdepth = (1-ret.depth)/ret.opacity if opt.camera.ndc else 1/(ret.depth/ret.opacity+eps)
                rgb_map = ret.rgb.view(-1,opt.H,opt.W,3).permute(0,3,1,2) # [B,3,H,W]
                invdepth_map = invdepth.view(-1,opt.H,opt.W,1).permute(0,3,1,2) # [B,1,H,W]
                torchvision_F.to_pil_image(rgb_map.cpu()[0]).save("{}/rgb_{}.png".format(novel_path,i))
                torchvision_F.to_pil_image(invdepth_map.cpu()[0]).save("{}/depth_{}.png".format(novel_path,i))
            # write videos
            print("writing videos...")
            rgb_vid_fname = "{}/novel_view_rgb.mp4".format(opt.output_path)
            depth_vid_fname = "{}/novel_view_depth.mp4".format(opt.output_path)
            os.system("ffmpeg -y -framerate 30 -i {0}/rgb_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,rgb_vid_fname))
            os.system("ffmpeg -y -framerate 30 -i {0}/depth_%d.png -pix_fmt yuv420p {1} >/dev/null 2>&1".format(novel_path,depth_vid_fname))

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.nerf = NeRF(opt)
        if opt.nerf.fine_sampling:
            self.nerf_fine = NeRF(opt)

    def forward(self,opt,var,mode=None):
        batch_size = len(var.idx)
        pose = self.get_pose(opt,var,mode=mode)
        # render images
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            # sample random rays for optimization
            if 'W' in var and 'H' in var: 
                var.ray_idx = torch.randperm(int(var.H)*int(var.W),device=opt.device)[:opt.nerf.rand_rays//batch_size]
                ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode, HW=(int(var.H), int(var.W)), idx=var.idx) # [B,N,3],[B,N,1]
            else: 
                var.ray_idx = torch.randperm(opt.H*opt.W,device=opt.device)[:opt.nerf.rand_rays//batch_size]
                ret = self.render(opt,pose,intr=var.intr,ray_idx=var.ray_idx,mode=mode, idx=var.idx) # [B,N,3],[B,N,1]
        else:
            # render full image (process in slices)
            if 'W' in var and 'H' in var: 
                ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode, HW=(int(var.H), int(var.W)), idx=var.idx) if opt.nerf.rand_rays else \
                    self.render(opt,pose,intr=var.intr,mode=mode, HW=(int(var.H), int(var.W)), idx=var.idx) # [B,HW,3],[B,HW,1]
            else:
                ret = self.render_by_slices(opt,pose,intr=var.intr,mode=mode, idx=var.idx) if opt.nerf.rand_rays else \
                    self.render(opt,pose,intr=var.intr,mode=mode, idx=var.idx) # [B,HW,3],[B,HW,1]
        var.update(ret)
        return var

    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        batch_size = len(var.idx)
        if 'W' in var and 'H' in var: W, H = int(var.W), int(var.H)
        else: W, H = opt.W, opt.H
        image = var.image.view(batch_size, 3, int(H*W)).permute(0,2,1)
        if opt.feature.encode:
            feat_gt = var.feat_gt.view(batch_size, opt.feature.dim, int(H*W)).permute(0,2,1)
        if opt.nerf.rand_rays and mode in ["train","test-optim"]:
            image = image[:,var.ray_idx]
            if opt.feature.encode:
                feat_gt = feat_gt[:,var.ray_idx.to(feat_gt.device)].to(var.ray_idx.device)
        coef = self.nerf.get_refine_coef(opt)
        if not opt.transient.encode and not opt.feature.encode:
            loss.rgb = self.MSE_loss(var.rgb,image)
            if opt.nerf.fine_sampling:
                loss.rgb_fine = self.MSE_loss(var.rgb_fine,image)
        elif not opt.transient.encode and opt.feature.encode:
            loss.feat = self.MSE_loss(var.feat,feat_gt) * (1-coef)
            loss.rgb = self.MSE_loss(var.rgb,image) * coef
            if opt.nerf.fine_sampling:
                loss.feat_fine = self.MSE_loss(var.feat_fine,feat_gt) * (1-coef)
                loss.rgb_fine = self.MSE_loss(var.rgb_fine,image) * coef
        elif opt.transient.encode and not opt.feature.encode:
            loss.rgb = self.MSE_loss(var.rgb,image) * (1-coef) 
            loss.static_rgb = self.MSE_loss(var.static_rgb,image) * coef
            if opt.nerf.fine_sampling:
                loss.rgb_fine = self.MSE_loss(var.rgb_fine,image) * (1-coef)
                loss.static_rgb_fine = self.MSE_loss(var.static_rgb_fine,image) * coef
        elif opt.transient.encode and opt.feature.encode:
            loss.feat = self.MSE_loss(var.feat,feat_gt) * (1-coef)
            loss.static_rgb = self.MSE_loss(var.static_rgb,image) * coef
            if opt.nerf.fine_sampling:
                loss.feat_fine = self.MSE_loss(var.feat_fine,feat_gt) * (1-coef)
                loss.static_rgb_fine = self.MSE_loss(var.static_rgb_fine,image) * coef
        mse = edict(coarse = self.MSE_loss(var.static_rgb,image))
        if opt.nerf.fine_sampling:
            mse.update(fine = self.MSE_loss(var.static_rgb_fine,image))
        if opt.feature.encode:
            mse.update(coarse_feat = self.MSE_loss(var.feat,feat_gt))
            if opt.nerf.fine_sampling:
                mse.update(fine_feat = self.MSE_loss(var.feat_fine,feat_gt))
        return loss, mse

    def get_pose(self,opt,var,mode=None):
        return var.pose

    def render(self,opt,pose,intr=None,ray_idx=None,mode=None, HW=None, idx=None):
        ret = edict() 
        batch_size = len(pose)
        center,ray = camera.get_center_and_ray(opt,pose,intr=intr, HW=HW) # [B,HW,3]
        while ray.isnan().any(): # TODO: weird bug, ray becomes NaN arbitrarily if batch_size>1, not deterministic reproducible
            center,ray = camera.get_center_and_ray(opt,pose,intr=intr, HW=HW) # [B,HW,3]
        if ray_idx is not None:
            # consider only subset of rays
            center,ray = center[:,ray_idx],ray[:,ray_idx]
        if opt.camera.ndc:
            # convert center/ray representations to NDC
            center,ray = camera.convert_NDC(opt,center,ray,intr=intr)    
        # render with main MLP
        depth_samples = self.sample_depth(opt,batch_size,num_rays=ray.shape[1]) # [B,HW,N,1]
        t_emb = self.embedding_t.weight[idx] if opt.transient.encode else None 
        samples = self.nerf.forward_samples(opt,center,ray,depth_samples,mode=mode,t_emb=t_emb)
        ret_composite = self.nerf.composite(opt,ray,samples,depth_samples)
        util.merge_edict(ret_composite, ret)
        # render with fine MLP from coarse MLP
        if opt.nerf.fine_sampling:
            with torch.no_grad():
                # resample depth acoording to coarse empirical distribution
                depth_samples_fine = self.sample_depth_from_pdf(opt,pdf=ret_composite.prob[...,0]) # [B,HW,Nf,1]
                depth_samples = torch.cat([depth_samples,depth_samples_fine],dim=2) # [B,HW,N+Nf,1]
                depth_samples = depth_samples.sort(dim=2).values
            samples_fine = self.nerf_fine.forward_samples(opt,center,ray,depth_samples,mode=mode,t_emb=t_emb)
            ret_composite_fine = self.nerf_fine.composite(opt,ray,samples_fine,depth_samples)
            util.merge_edict(ret_composite_fine, ret, postfix="_fine")
        return ret

    def render_by_slices(self,opt,pose,intr=None,mode=None, HW=None, idx=None):
        ret_all = edict()
        if HW is None: H, W = opt.H, opt.W
        else: H, W = HW
        # render the image by slices for memory considerations
        for c in range(0, H*W,opt.nerf.rand_rays):
            ray_idx = torch.arange(c,min(c+opt.nerf.rand_rays,H*W),device=opt.device)
            ret = self.render(opt,pose,intr=intr,ray_idx=ray_idx,mode=mode, HW=HW, idx=idx) # [B,R,3],[B,R,1]
            for k in ret:
                if k not in ret_all:
                    ret_all[k] = []
                ret_all[k].append(ret[k])
        # group all slices of images
        for k in ret_all: ret_all[k] = torch.cat(ret_all[k],dim=1)
        return ret_all

    def sample_depth(self,opt,batch_size,num_rays):
        depth_min,depth_max = opt.nerf.depth.range
        num_rays = num_rays
        rand_samples = torch.rand(batch_size,num_rays,opt.nerf.sample_intvs,1,device=opt.device) if opt.nerf.sample_stratified else 0.5
        rand_samples += torch.arange(opt.nerf.sample_intvs,device=opt.device)[None,None,:,None].float() # [B,HW,N,1]
        depth_samples = rand_samples/opt.nerf.sample_intvs*(depth_max-depth_min)+depth_min # [B,HW,N,1]
        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]
        return depth_samples

    def sample_depth_from_pdf(self,opt,pdf):
        depth_min,depth_max = opt.nerf.depth.range
        # get CDF from PDF (along last dimension)
        cdf = pdf.cumsum(dim=-1) # [B,HW,N]
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],dim=-1) # [B,HW,N+1]
        # take uniform samples
        grid = torch.linspace(0,1,opt.nerf.sample_intvs_fine+1,device=opt.device) # [Nf+1]
        unif = 0.5*(grid[:-1]+grid[1:]).repeat(*cdf.shape[:-1],1) # [B,HW,Nf]
        idx = torch.searchsorted(cdf,unif,right=True) # [B,HW,Nf] \in {1...N}
        # inverse transform sampling from CDF
        depth_bin = torch.linspace(depth_min,depth_max,opt.nerf.sample_intvs+1,device=opt.device) # [N+1]
        depth_bin = depth_bin.repeat(*cdf.shape[:-1],1) # [B,HW,N+1]
        depth_low = depth_bin.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        depth_high = depth_bin.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        cdf_low = cdf.gather(dim=2,index=(idx-1).clamp(min=0)) # [B,HW,Nf]
        cdf_high = cdf.gather(dim=2,index=idx.clamp(max=opt.nerf.sample_intvs)) # [B,HW,Nf]
        # linear interpolation
        t = (unif-cdf_low)/(cdf_high-cdf_low+1e-8) # [B,HW,Nf]
        depth_samples = depth_low+t*(depth_high-depth_low) # [B,HW,Nf]
        return depth_samples[...,None] # [B,HW,Nf,1]

class NeRF(torch.nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        input_3D_dim = 3+6*opt.arch.posenc.L_3D if opt.arch.posenc else 3
        if opt.nerf.view_dep:
            input_view_dim = 3+6*opt.arch.posenc.L_view if opt.arch.posenc else 3
        # point-wise feature
        self.mlp_xyz = torch.nn.ModuleList()
        xyz_dims = util.get_layer_dims(opt.arch.layers_feat)
        for li,(k_in,k_out) in enumerate(xyz_dims):
            if li==0: k_in = input_3D_dim
            if li in opt.arch.skip: k_in += input_3D_dim
            if li==len(xyz_dims)-1 and not opt.transient.encode and not opt.feature.encode: k_out += 1
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="first" if li==len(xyz_dims)-1 else None)
            self.mlp_xyz.append(linear)
        if opt.transient.encode or opt.feature.encode:
            # Our implementation has one additional layer
            self.mlp_xyz_final = torch.nn.Linear(k_out,k_out)
            # Our implementation has specific layer for density
            self.mlp_density = torch.nn.Sequential(
                torch.nn.Linear(k_out, 1),
                torch.nn.Softplus()
            )
        if opt.feature.encode:
            self.mlp_feat = torch.nn.Linear(k_out, opt.feature.dim)
        # RGB prediction
        self.mlp_rgb = torch.nn.ModuleList()
        rgb_dims = util.get_layer_dims(opt.arch.layers_rgb)
        xyz_dim = opt.arch.layers_feat[-1]
        for li,(k_in,k_out) in enumerate(rgb_dims):
            if li==0: 
                if not opt.feature.encode:
                    k_in = xyz_dim+(input_view_dim if opt.nerf.view_dep else 0)
                else:
                    k_in = opt.feature.dim+(input_view_dim if opt.nerf.view_dep else 0)
            linear = torch.nn.Linear(k_in,k_out)
            if opt.arch.tf_init:
                self.tensorflow_init_weights(opt,linear,out="all" if li==len(rgb_dims)-1 else None)
            self.mlp_rgb.append(linear)
        # transient feature
        if opt.transient.encode:
            self.mlp_xyz_t = torch.nn.ModuleList()
            xyz_t_dim = xyz_dim//2
            for i in range(opt.transient.num_layers):
                if i==0: k_in = xyz_dim + opt.transient.size
                else: k_in = xyz_t_dim
                k_out = xyz_t_dim
                linear = torch.nn.Linear(k_in,k_out)
                self.mlp_xyz_t.append(linear)
            # density prediction
            self.mlp_density_t = torch.nn.Sequential(
                torch.nn.Linear(k_out, 1),
                torch.nn.Softplus()
            ) 
            # RGB prediction
            if not opt.feature.encode:
                self.mlp_rgb_t = torch.nn.Sequential(
                    torch.nn.Linear(k_out, 3),
                    torch.nn.Sigmoid()
                ) 
            else:
                self.mlp_feat_t = torch.nn.Linear(k_out, opt.feature.dim)

    def tensorflow_init_weights(self,opt,linear,out=None):
        # use Xavier init instead of Kaiming init
        relu_gain = torch.nn.init.calculate_gain("relu") # sqrt(2)
        if out=="all":
            torch.nn.init.xavier_uniform_(linear.weight)
        elif out=="first":
            torch.nn.init.xavier_uniform_(linear.weight[:1])
            torch.nn.init.xavier_uniform_(linear.weight[1:],gain=relu_gain)
        else:
            torch.nn.init.xavier_uniform_(linear.weight,gain=relu_gain)
        torch.nn.init.zeros_(linear.bias)

    def forward(self,opt,points_3D,ray_unit=None,mode=None,t_emb=None): # [B,...,3]
        if opt.arch.posenc:
            points_enc = self.positional_encoding(opt,points_3D,L=opt.arch.posenc.L_3D)
            points_enc = torch.cat([points_3D,points_enc],dim=-1) # [B,...,6L+3]
        else: points_enc = points_3D
        xyz = points_enc
        if not (opt.transient.encode or opt.feature.encode): # Original BARF
            # extract coordinate-based features (with density)
            for li,layer in enumerate(self.mlp_xyz):
                if li in opt.arch.skip: xyz = torch.cat([xyz,points_enc],dim=-1)
                xyz = layer(xyz)
                if li==len(self.mlp_xyz)-1:
                    density = xyz[...,0]
                    if opt.nerf.density_noise_reg and mode=="train":
                        density += torch.randn_like(density)*opt.nerf.density_noise_reg
                    density_activ = getattr(torch_F,opt.arch.density_activ) # relu_,abs_,sigmoid_,exp_....
                    density = density_activ(density)
                    xyz = xyz[...,1:] 
                xyz = torch_F.relu(xyz)
            if opt.nerf.view_dep:
                assert(ray_unit is not None)
                if opt.arch.posenc:
                    ray_enc = self.positional_encoding(opt,ray_unit,L=opt.arch.posenc.L_view)
                    ray_enc = torch.cat([ray_unit,ray_enc],dim=-1) # [B,...,6L+3]
                else: ray_enc = ray_unit
                rgb = torch.cat([xyz,ray_enc],dim=-1)
            else:
                rgb = xyz
            # predict RGB values
            for li,layer in enumerate(self.mlp_rgb):
                rgb = layer(rgb)
                if li!=len(self.mlp_rgb)-1:
                    rgb = torch_F.relu(rgb)
            rgb = rgb.sigmoid_() # [B,...,3]
            return edict(rgb=rgb, density=density)
        else: # Ours
            ret = edict()
            # extract coordinate-based features (with density)
            for li,layer in enumerate(self.mlp_xyz):
                if li in opt.arch.skip: xyz = torch.cat([xyz,points_enc],dim=-1)
                xyz = layer(xyz)
                xyz = torch_F.relu(xyz)
            density = self.mlp_density(xyz).squeeze(-1)
            ret.update(density=density)
            xyz = self.mlp_xyz_final(xyz)
            # extract transient features (with density)
            if opt.transient.encode:
                while t_emb.dim() < xyz.dim():
                    t_emb = t_emb.unsqueeze(1)
                t_emb = t_emb.repeat(*([1] + list(xyz.shape[1:-1]) + [1]))
                xyz_t = torch.cat([xyz,t_emb],dim=-1)
                for li,layer in enumerate(self.mlp_xyz_t):
                    xyz_t = layer(xyz_t)
                    xyz_t = torch_F.relu(xyz_t)
                # predict transient density
                density_t = self.mlp_density_t(xyz_t).squeeze(-1)
                ret.update(density_t=density_t)
            if opt.feature.encode: # rgb + feature
                feat = self.mlp_feat(xyz)
                ret.update(feat=feat)
                if opt.transient.encode:
                    feat_t = self.mlp_feat_t(xyz_t)
                    ret.update(feat_t=feat_t)
                rgb = feat
            else:
                rgb = xyz
            if opt.nerf.view_dep:
                assert(ray_unit is not None)
                if opt.arch.posenc:
                    ray_enc = self.positional_encoding(opt,ray_unit,L=opt.arch.posenc.L_view)
                    ray_enc = torch.cat([ray_unit,ray_enc],dim=-1) # [B,...,6L+3]
                else: ray_enc = ray_unit
                rgb = torch.cat([rgb,ray_enc.detach()],dim=-1)
            # predict RGB values 
            for li,layer in enumerate(self.mlp_rgb):
                rgb = layer(rgb)
                if li!=len(self.mlp_rgb)-1:
                    rgb = torch_F.relu(rgb)
            rgb = rgb.sigmoid_() # [B,...,3]
            ret.update(rgb=rgb)
            if not opt.feature.encode:
                rgb_t = self.mlp_rgb_t(xyz_t)
                ret.update(rgb_t=rgb_t)
            return ret
        
    def forward_samples(self,opt,center,ray,depth_samples,mode=None, t_emb=None):
        points_3D_samples = camera.get_3D_points_from_depth(opt,center,ray,depth_samples,multi_samples=True) # [B,HW,N,3]
        if opt.nerf.view_dep:
            ray_unit = torch_F.normalize(ray,dim=-1) # [B,HW,3]
            ray_unit_samples = ray_unit[...,None,:].expand_as(points_3D_samples) # [B,HW,N,3]
        else: ray_unit_samples = None
        return self.forward(opt,points_3D_samples,ray_unit=ray_unit_samples,mode=mode,t_emb=t_emb) # [B,HW,N],[B,HW,N,3]

    def composite(self,opt,ray,samples,depth_samples):
        ret = edict()
        ray_length = ray.norm(dim=-1,keepdim=True) # [B,HW,1]
        # volume rendering: compute probability (using quadrature)
        depth_intv_samples = depth_samples[...,1:,0]-depth_samples[...,:-1,0] # [B,HW,N-1]
        depth_intv_samples = torch.cat([depth_intv_samples,torch.empty_like(depth_intv_samples[...,:1]).fill_(1e10)],dim=2) # [B,HW,N]
        dist_samples = depth_intv_samples*ray_length # [B,HW,N]
        sigma_delta = samples.density*dist_samples # [B,HW,N]\
        alpha = 1-(-sigma_delta).exp_() # [B,HW,N]
        if opt.transient.encode:
            sigma_t_delta = samples.density_t*dist_samples # [B,HW,N]
            alpha_t = 1-(-sigma_t_delta).exp_() # [B,HW,N]
            alpha_sum = 1-(-(sigma_delta + sigma_t_delta)).exp_()
        else:
            alpha_sum = alpha
        static_T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
        if opt.transient.encode:
            composite_T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]+sigma_t_delta[...,:-1]],dim=2).cumsum(dim=2)).exp_() # [B,HW,N]
            transient_weight = (composite_T*alpha_t)[...,None]
        else:
            composite_T = static_T
        static_only_weight = (static_T*alpha)[...,None]
        static_weight = (composite_T*alpha)[...,None]
        static_prob = (static_T*alpha)[...,None] # [B,HW,N,1]
        prob = (composite_T*alpha_sum)[...,None] # [B,HW,N,1]
        opacity = prob.sum(dim=2) # [B,HW,1]
        # integrate RGB and depth weighted by probability
        static_depth = (depth_samples*static_prob).sum(dim=2)
        depth = (depth_samples*prob).sum(dim=2) # [B,HW,1]
        ret.update(opacity=opacity, prob=prob, depth=depth, static_depth=static_depth)
        static_rgb = (samples.rgb*static_only_weight).sum(dim=2)
        if opt.nerf.setbg_opaque:
            static_rgb = static_rgb+opt.data.bgcolor*(1-opacity)
        if opt.transient.encode and not opt.feature.encode:
            rgb = (samples.rgb*static_weight + samples.rgb_t*transient_weight).sum(dim=2) # [B,HW,3]
        else:
            rgb = static_rgb
        if opt.nerf.setbg_opaque:
            rgb = rgb+opt.data.bgcolor*(1-opacity)
        ret.update(rgb=rgb, static_rgb=static_rgb)
        if opt.feature.encode:
            static_feat = (samples.feat*static_only_weight).sum(dim=2) # [B,HW,3]
            if opt.transient.encode:
                feat = (samples.feat*static_weight + samples.feat_t*transient_weight).sum(dim=2) # [B,HW,3]
            else:
                feat = static_feat
            ret.update(feat=feat)
        return ret

    def positional_encoding(self,opt,input,L): # [B,...,N]
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,N,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
        return input_enc
