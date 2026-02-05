#!/bin/bash

set -e

export TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64/:$CONDA_PREFIX/bin/:$CONDA_PREFIX/lib/:$CONDA_PREFIX/lib/stub:$LD_LIBRARY_PATH"

export CUDA_CACHE_PATH=/tmp/$USER/$HOSTNAME/


python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="bell" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="angel" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="cat" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="horse" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="luyu" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="potion" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="tbell" baseline
python scripts/glossy_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="teapot" baseline

# python scripts/real_eval.py --skip_train --skip_render --skip_metrics --render_path --scene="gardenspheres" baseline_r4
python scripts/real_eval.py --skip_train --skip_render --skip_metrics --render_path --scene="sedan" baseline_r8
python scripts/real_eval.py --skip_train --skip_render --skip_metrics --render_path --scene="toycar" baseline_r4


# Redundant. Already rendered in test.
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="ball" baseline
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="car" baseline
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="coffee" baseline
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="helmet" baseline
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="teapot" baseline
# python scripts/shiny_eval.py --skip_train --skip_render --skip_metrics --skip_relight --render_path --scene="toaster" baseline