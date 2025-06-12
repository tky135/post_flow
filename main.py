from rectified_flow.datasets.toy_gmm import TwoPointGMM, CircularGMM, LowDimData
from rectified_flow.models.toy_mlp import MLPVelocity, GMFlowMLP2DDenoiser, VNetD
from rectified_flow.rectified_flow import RectifiedFlow, ScoringRule, GMFlow, HRF
from rectified_flow.utils import set_seed, plot_traj
from rectified_flow.samplers import EulerSampler
from datasets.distribution import Gaussian, Moon, FlowData
from rectified_flow.datasets.img_dataset import get_datalooper
from rectified_flow.utils import visualize_2d_trajectories_plotly
from scipy.stats import wasserstein_distance
from utils.visualize import visualize_combined
import torch
import argparse
import matplotlib.pyplot as plt
from ipdb import iex
from omegaconf import OmegaConf
import ot
import os
import sys
import tqdm
import numpy as np
import importlib
import torchvision

def cal_wd(x_1_gt, x_1_hat):
    assert x_1_gt.shape == x_1_hat.shape, "x_1_gt and x_1_hat must have the same shape"
    assert len(x_1_gt.shape) == 2, "x_1_gt and x_1_hat must be 2D tensors"
    
    if x_1_gt.shape[1] == 1:  # 1D case
        return wasserstein_distance(x_1_gt.squeeze(-1).cpu().numpy(), x_1_hat.squeeze(-1).cpu().numpy())
    elif x_1_gt.shape[1] == 2:  # 2D case
        return ot.sliced_wasserstein_distance(x_1_gt, x_1_hat, seed=1)
def import_str(string: str):
    """ Import a python module given string paths

    Args:
        string (str): The given paths

    Returns:
        Any: Imported python module / object
    """
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)

def get_object(cfg):
    if hasattr(cfg, 'args'):
        return import_str(cfg.type)(**cfg.args)
    else:
        return import_str(cfg.type)()
def build_dataset(data_cfg):
    """
    Return: source distribution, target distribution, dataloader
    """
    source_dist = get_object(data_cfg.source)
    target_dist = get_object(data_cfg.target)
    dataset = FlowData(source=source_dist, target=target_dist, light_weight=data_cfg.light_weight, data_shape=data_cfg.data_shape)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1 if data_cfg.light_weight else data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        prefetch_factor=data_cfg.prefetch_factor,
    )

    return source_dist, target_dist, iter(dataloader)

def build_model(model_cfg, source_distribution="normal", device="cuda"):
    net = get_object(model_cfg.net).to(device)
    fm_model = import_str(model_cfg.method.type)(
        velocity_field=net,
        source_distribution=source_distribution,
        device=device,
        data_shape=model_cfg.data_shape,
        **model_cfg.method.args
    )
    model_size = 0
    for param in net.parameters():
        model_size += param.data.nelement()
    print(f"Model params number: {model_size}")
    print("Model params: %.2f M" % (model_size / 1000 / 1000))
    
    return net, fm_model
@iex
def main(args):


    # setup
    logdir = os.path.join(args.logdir, args.exp_name)
    ckptdir = os.path.join(logdir, "checkpoints")
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)

    cfg = OmegaConf.load(args.config_file)
    cfg_cli = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(cfg, cfg_cli)
    
    cfg.data_shape = cfg.data_shape
    cfg.model.data_shape = cfg.data_shape
    cfg.data.data_shape = cfg.data_shape

    pi_0, pi_1, dataiter = build_dataset(cfg.data)

    # network
    
    net, fm_model = build_model(cfg.model, source_distribution=pi_0, device=device)
    method = cfg.model.method.type.split('.')[-1]  # Extract the last part of the method type string
    
    # training 
    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (min(step, 5000) / 5000))
    
    
    
    
    cur_step = 0
    # load from checkpoint
    ckpt_list = os.listdir(ckptdir)
    if args.resume and len(ckpt_list) > 0:
        ckpt = sorted(ckpt_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        print(f"loading {ckpt}")
        ckpt = torch.load(os.path.join(ckptdir, ckpt), weights_only=True)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        cur_step = ckpt['step']
    elif args.resume_from is not None:
        print(f"loading {args.resume_from}")
        ckpt = torch.load(args.resume_from, weights_only=True)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        cur_step = ckpt['step']
    
    
    for step in tqdm.tqdm(range(cur_step, 50000 + 1)):
        x_0, x_1 = next(dataiter)
        if cfg.data.light_weight:
            x_0, x_1 = x_0.squeeze(0), x_1.squeeze(0)
        x_0, x_1 = x_0.to(device), x_1.to(device)
        optimizer.zero_grad()
        loss = fm_model.get_loss(x_0, x_1)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            if len(cfg.data_shape) == 1:
                print(f"Epoch {step}, Loss: {loss.item()}", end='\t')
                euler_sampler_1rf_unconditional = EulerSampler(
                    fm_model,
                    num_steps=10
                )
                x_0_sample = pi_0.sample([10000]).to(device)
                x_1_gt = pi_1.sample([10000]).to(device)
                traj_upper = euler_sampler_1rf_unconditional.sample_loop(x_0=x_0_sample).trajectories
                x_1_hat = traj_upper[-1]
                wd = cal_wd(x_1_gt, x_1_hat)
                print("SWD:", wd.item())
            else:
                print(f"Epoch {step}, Loss: {loss.item()}")
                STEPS = 10
                SAMPLES = 10
                # display images
                euler_sampler_1rf_unconditional = EulerSampler(
                    fm_model,
                    num_steps=STEPS
                )
                x_0_sample = pi_0.sample([SAMPLES]).to(device)
                traj_upper = euler_sampler_1rf_unconditional.sample_loop(x_0=x_0_sample).trajectories
                torchvision.utils.save_image(
                    torch.cat(traj_upper, dim=0),
                    os.path.join(logdir, f"img_{method}_{step}_iters.png"),
                    nrow=SAMPLES,
                    normalize=True,
                    value_range=(-1, 1)
                )
        if step % 10000 == 0:
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step + 1,
                
            }, os.path.join(ckptdir, f"{args.exp_name}_step_{step}.pt"))



    SAMPLE_NUM = 1000000
    STEP_NUM = [1, 2, 5, 10, 20, 50, 100]
    for step_num in STEP_NUM:
        euler_sampler_1rf_unconditional = EulerSampler(
            fm_model,
            num_steps=step_num
        )
        x_0_sample = pi_0.sample(SAMPLE_NUM).to(device)
        x_1_sample = pi_1.sample(SAMPLE_NUM).to(device)
        traj_upper = euler_sampler_1rf_unconditional.sample_loop(x_0=x_0_sample).trajectories

        x_1_hat = traj_upper[-1]
        x_1_gt = x_1_sample
        if torch.isnan(x_1_hat).any() or torch.isinf(x_1_hat).any():
            import ipdb ; ipdb.set_trace()
        wd = cal_wd(x_1_gt, x_1_hat)
        plot_traj(torch.stack(traj_upper, dim=0), distance=wd, title=f"Method: {method}, {step_num} steps", traj_dir=logdir, file_name=f"2d_{method}_{step_num}step.png")


        xt = x_1_hat
        plt.figure()
        if tuple(cfg.data_shape) == (2,):
            plt.scatter(x_0_sample[:10000, 0].cpu().numpy(), x_0_sample[:10000, 1].cpu().numpy(), c="#1f77b4", label="Source", alpha=0.25, s=3)
            plt.scatter(x_1_sample[:10000, 0].cpu().numpy(), x_1_sample[:10000, 1].cpu().numpy(), c="#ff7f0e", label="Target", alpha=0.25, s=3)
            plt.scatter(xt[:10000,0].cpu().numpy(), xt[:10000,1].cpu().numpy(), c="#2ca02c", label=f'Gen SWD={wd:.3f}', alpha=0.25, s=3)
        elif tuple(cfg.data_shape) == (1,):
            bins = np.linspace(-2, 2, 201)
            plt.hist(x_1_sample[:, 0].cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Target')
            plt.hist(xt.cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Gen WD={wd:.3f}')
        else:
            raise Exception("Unsupported data shape for visualization")
    
        plt.title(f"Method: {method}, {step_num} steps", fontsize=20)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axis('off')
        
        plt.legend(fontsize=16, framealpha=0.5, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, f"2d_dist_{method}_{step_num}_step.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sliced Wasserstein distance between x_1_hat and x_1_gt: {wd:.4f}")

    # visualize velocity distribution for 1D case
    if tuple(cfg.data_shape) == (1,):
        euler_sampler_1rf_unconditional = EulerSampler(
            fm_model,
            num_steps=100
        )
        x_0_sample = pi_0.sample(1).to(device) * 0.0
        x_1_sample = pi_1.sample(1).to(device)
        traj_upper = euler_sampler_1rf_unconditional.sample_loop(x_0=x_0_sample).trajectories

        BATCH_SIZE = 10000
        v_T_N, x_1_hat_T_N = [], []
        for i in range(len(traj_upper)):
            xt = traj_upper[i]
            t = i / (len(traj_upper) - 1)
            xt_batch = xt.repeat(BATCH_SIZE, 1)
            vt_batch = fm_model.get_velocity(xt_batch, t)
            x_1_hat_batch = xt_batch + vt_batch * (1 - t)  # Forward Euler step
            v_T_N.append(vt_batch.squeeze().cpu())
            x_1_hat_T_N.append(x_1_hat_batch.squeeze().cpu())
        v_T_N = torch.stack(v_T_N, dim=0)
        x_1_hat_T_N = torch.stack(x_1_hat_T_N, dim=0)
        gt_samples = pi_1.sample(BATCH_SIZE)
        visualize_combined(data_tensor=v_T_N[:, :-1],
                           trajs={f"{method}": traj_upper},
                           D1_gt_samples=gt_samples,
                           save_path=os.path.join(logdir, f"p(v|t)_{method}.html"),
                           x_range=(-2, 2)
                           )
        visualize_combined(data_tensor=x_1_hat_T_N[:, :-1],
                           trajs={f"{method}": traj_upper},
                           D1_gt_samples=gt_samples,
                           save_path=os.path.join(logdir, f"p(x|t)_{method}.html"),
                           x_range=(-2, 2)
                           )





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, help="Path to the config file")
    parser.add_argument("--logdir", type=str, default="log", help="Directory to save logs and results")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name for logging")
    parser.add_argument("--resume", action='store_true', help="Resume from the last checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to the checkpoint to resume from")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args)