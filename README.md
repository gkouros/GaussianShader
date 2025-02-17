<p align="center">

  <h1 align="center">GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces</h1>
  <p align="center">
    <a href="https://github.com/Asparagus15">Yingwenqi Jiang</a>,
    <a href="https://github.com/donjiaking">Jiadong Tu</a>,
    <a href="https://liuyuan-pal.github.io/">Yuan Liu</a>,
    <a href="https://gaoxifeng.github.io/">Xifeng Gao</a>,
    <a href="https://www.xxlong.site/">Xiaoxiao Long*</a>,
    <a href="https://www.cs.hku.hk/people/academic-staff/wenping">Wenping Wang</a>,
    <a href="https://yuexinma.me/aboutme.html">Yuexin Ma*</a>

  </p>
    <p align="center">
    *Corresponding authors

  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2311.17977">Paper</a> | <a href="https://asparagus15.github.io/GaussianShader.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

## Introduction
The advent of neural 3D Gaussians has recently brought about a revolution in the field of neural rendering, facilitating the generation of high-quality renderings at real-time speeds. However, the explicit and discrete representation encounters challenges when applied to scenes featuring reflective surfaces. In this paper, we present **GaussianShader**, a novel method that applies a simplified shading function on 3D Gaussians to enhance the neural rendering in scenes with reflective surfaces while preserving the training and rendering efficiency.

<p align="center">
  <a href="">
    <img src="assets/relit.gif" alt="Relit" width="95%">
  </a>
</p>
<p align="center">
  GaussianShader maintains real-time rendering speed and renders high-fidelity images for both general and reflective surfaces. GaussianShader enables free-viewpoint rendering objects under distinct lighting environments.
</p>
<br>

<p align="center">
  <a href="">
    <img src="./assets/pipeline.png" alt="Pipeline" width="95%">
  </a>
</p>
<p align="center">
  GaussianShader initiates with the neural 3D Gaussian spheres that integrate both conventional attributes and the newly introduced
  shading attributes to accurately capture view-dependent appearances. We incorporate a differentiable environment lighting map to simulate
  realistic lighting. The end-to-end training leads to a model that reconstructs both reflective and diffuse surfaces, achieving high material
  and lighting fidelity.
</p>
<br>

## Installation
Provide installation instructions for your project. Include any dependencies and commands needed to set up the project.


```shell
# Clone the repository
git clone https://github.com/Asparagus15/GaussianShader.git
cd GaussianShader

# Install dependencies
conda env create --file environment.yml
conda activate gaussian_shader
```

For cuda problems reinstall all cuda modules with fixed version:
```bash
mamba install -c nvidia cuda-toolkit=11.8 cuda-memcheck=11.8 gds-tools=1.9.1.3 cuda-documentation=11.8 cuda-nvml-dev=11.8 cuda-cccl=11.8 cuda-driver-dev=11.8 cuda-nvrtc-dev=11.8 cuda-profiler-api=11.8 cuda-gdb=11.8 cuda-nvdisasm=11.8 cuda-nvprof=11.8 cuda-sanitizer-api=11.8 cuda-cuobjdump=11.8 cuda-cuxxfilt=11.8 cuda-nvcc=11.8 cuda-nvprune=11.8 cuda-nsight=11.8 cuda-cudart-dev=11.8 cuda-command-line-tools=11.8 cuda-compiler=11.8 cuda-libraries-dev=11.8 cuda-visual-tools=11.8 cuda-tools=11.8 cuda-nsight-compute=11.8 cuda-nvvp=11.8

mamba install -c nvidia cuda-toolkit=12.6 gds-tools=1.9.1.3 cuda-nvml-dev=12.6 cuda-cccl=12.6 cuda-driver-dev=12.6 cuda-nvrtc-dev=12.6 cuda-profiler-api=12.6 cuda-gdb=12.6 cuda-nvdisasm=12.6 cuda-nvprof=12.6 cuda-sanitizer-api=12.6 cuda-cuobjdump=12.6 cuda-cuxxfilt=12.6 cuda-nvcc=12.6 cuda-nvprune=12.6 cuda-nsight=12.6 cuda-cudart-dev=12.6 cuda-command-line-tools=12.6 cuda-compiler=12.6 cuda-libraries-dev=12.6 cuda-visual-tools=12.6 cuda-tools=12.6 cuda-nsight-compute=12.6 cuda-nvvp=12.6
```

For -lcuda not found error look at: https://github.com/NVlabs/tiny-cuda-nn/issues/183
and add symlinks from /users/visics/gkouros/miniforge3/envs/igs2/lib/stubs/lcuda
to /users/visics/gkouros/miniforge3/envs/igs2/lib

For missing nvdiffrast
```
pip install git+https://github.com/NVlabs/nvdiffrast/
```

## Running
Download the [example data](https://drive.google.com/file/d/1bSv0soQtjbRj9S9Aq9uQ27EW4wwY--6q/view?usp=sharing) and put it to the ``data`` folder. Execute the optimizer using the following command:
```shell
python train.py -s data/horse_blender --eval -m output/horse_blender -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512
```

## Rendering
```shell
python render.py -m output/horse_blender --brdf_dim 0 --sh_degree -1 --brdf_mode envmap --brdf_env 512
```

## Dataset
We mainly evaluate our method on [NeRF Synthetic](https://github.com/bmild/nerf), [Tanks&Temples](https://www.tanksandtemples.org), [Shiny Blender](https://github.com/google-research/multinerf) and [Glossy Synthetic](https://github.com/liuyuan-pal/NeRO). You can use ``nero2blender.py`` to convert the Glossy Synthetic data into Blender format.

## More features
The repo is still being under construction, thanks for your patience.
- [ ] Arguments explanation.
- [ ] Residual color training code.

## Acknowledgement
We have intensively borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Ref-NeRF](https://github.com/google-research/multinerf)
- [nvdiffrec](https://github.com/NVlabs/nvdiffrec)
- [Point-NeRF](https://github.com/Xharlie/pointnerf)

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@article{jiang2023gaussianshader,
  title={GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces},
  author={Jiang, Yingwenqi and Tu, Jiadong and Liu, Yuan and Gao, Xifeng and Long, Xiaoxiao and Wang, Wenping and Ma, Yuexin},
  journal={arXiv preprint arXiv:2311.17977},
  year={2023}
}
```
