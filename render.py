#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs, name
from gaussian_renderer import render, render_lighting, render_lighting2
import torchvision
from utils.general_utils import safe_state
from utils.render_utils import generate_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import apply_depth_colormap
from utils.general_utils import get_minimum_axis
from scene.NVDIFFREC.util import save_image_raw, gamma_tonemap, load_image_raw, save_image, save_image_raw
from scene.NVDIFFREC.light import save_env_map, save_env_map2
import numpy as np

def render_lightings(model_path, name, iteration, gaussians, sample_num):
    lighting_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    makedirs(lighting_path, exist_ok=True)
    # sampled_indicies = torch.randperm(gaussians.get_xyz.shape[0])[:sample_num]
    sampled_indicies = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long)[:sample_num]
    for sampled_index in tqdm(sampled_indicies, desc="Rendering lighting progress"):
        lighting = render_lighting(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(lighting, os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".png"))
        save_image_raw(os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + ".hdr"), lighting.permute(1,2,0).detach().cpu().numpy())
        lighting2 = render_lighting2(gaussians, sampled_index=sampled_index)
        torchvision.utils.save_image(lighting2, os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + "2.png"))
        save_image_raw(os.path.join(lighting_path, '{0:05d}'.format(sampled_index) + "2.hdr"), lighting2.permute(1,2,0).detach().cpu().numpy())

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, misc=True, rescale=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()

        render_pkg = render(view, gaussians, pipeline, background, debug=True)
        torch.cuda.synchronize()

        gt = view.original_image[0:3, :, :]
        rend = render_pkg["render"]

        if rescale:
            mask = view.gt_alpha_mask.bool().to(rend.device)
            # mask = render_pkg["alpha"]
            gt_mean = gt[:3, mask.squeeze()].mean(axis=1)
            rend_mean = rend[:3, mask.squeeze()].mean(axis=1)
            factor = gt_mean / rend_mean
            rend = factor[:, None, None] * rend * mask + (1 - mask.float())

        torchvision.utils.save_image(rend, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if "traj" not in name:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        # always save normals
        for k in render_pkg.keys():
            if "normal" in k:
                save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
                makedirs(save_path, exist_ok=True)
                render_pkg[k] = 0.5 + (0.5*render_pkg[k])

        # save misc features if requested
        if misc:
            for k in render_pkg.keys():
                if render_pkg[k].dim()<3 or k=="render" or k=="delta_normal_norm":
                    continue
                save_path = os.path.join(model_path, name, "ours_{}".format(iteration), k)
                makedirs(save_path, exist_ok=True)
                if k == "alpha":
                    render_pkg[k] = apply_depth_colormap(render_pkg["alpha"][0][...,None], min=0., max=1.).permute(2,0,1)
                elif k == "depth":
                    render_pkg[k] = apply_depth_colormap(-render_pkg["depth"][0][...,None]).permute(2,0,1)
                # elif k == "reflvec":
                #     render_pkg[k] = 0.5 + (0.5*render_pkg[k]*mask)
                torchvision.utils.save_image(render_pkg[k], os.path.join(save_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_dim, pipeline.brdf_mode, dataset.brdf_envmap_res)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        if pipeline.brdf:
            render_lightings(dataset.model_path, "lighting", scene.loaded_iter, gaussians, sample_num=1)

        if args.render_path and not args.relight_envmap_path:
            os.makedirs("traj", exist_ok=True)
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=240)
            render_set(dataset.model_path, "traj", scene.loaded_iter, cam_traj, gaussians, pipeline, background)

        if args.relight_gt_path and args.relight_envmap_path:
            if not args.render_path:
                dataset.source_path = args.relight_gt_path
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, relight=(not args.render_path))

            # get relighting paths
            relight_envmap = load_image_raw(args.relight_envmap_path)
            dirname = "relight" if not args.render_path else "traj_relight"
            relight_dir = os.path.join(dirname, args.relight_envmap_path.split('/')[-1])

            # load relight envmap and save
            os.makedirs(os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", 'envmaps'), exist_ok=True)
            save_image_raw(os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", 'envmaps', 'relight_envmap.hdr'), relight_envmap)
            save_image(os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", 'envmaps', 'relight_envmap.png'), relight_envmap)
            relight_envmap_tonemapped = gamma_tonemap(relight_envmap)
            save_image_raw(os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", 'envmaps', "relight_envmap_tonemapped.hdr"), relight_envmap_tonemapped)
            save_image(os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", 'envmaps', "relight_envmap_tonemapped.png"), relight_envmap_tonemapped)

            # replace envmap
            tonemap = lambda x: np.roll(gamma_tonemap(x), x.shape[1]//4, axis=1)
            gaussians.load_env_map(args.relight_envmap_path, tonemap=tonemap, rotate=True)

            # save relighting envmaps for sanity check
            brdf_mlp_path = os.path.join(dataset.model_path, relight_dir, f"ours_{scene.loaded_iter}", "envmaps", "brdf_mlp.hdr")
            save_env_map(brdf_mlp_path, gaussians.brdf_mlp)
            save_env_map2(brdf_mlp_path.replace(".hdr", "2.hdr"), gaussians.brdf_mlp)

            if args.render_path:
                n_frames = 240
                # import pdb; pdb.set_trace()
                cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
            else:
                cam_traj = scene.getTestCameras()

            # render relighted views
            render_set(dataset.model_path, relight_dir, scene.loaded_iter, cam_traj, gaussians, pipeline, background, misc=False, rescale=args.rescale_relighted)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--relight_envmap_path", default="", help="The envmap to use to relight the scene")
    parser.add_argument("--relight_gt_path", default="", help="The relighted dataset to compare against")
    parser.add_argument("--rescale_relighted", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)