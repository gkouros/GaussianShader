# training scripts for the shiny blender datasets
# this script is adopted from GOF
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_nerf_synthetic.py
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
import argparse

parser = argparse.ArgumentParser(description="Training script parameters")
parser.add_argument('out', help='Input experiment name')
parser.add_argument('--skip_train', help='Skip training step', action='store_true')
parser.add_argument('--skip_render', help='Skip rendering step', action='store_true')
parser.add_argument('--skip_metrics', help='Skip rendering step', action='store_true')

args = parser.parse_args()

dataset = "ref_shiny"
scenes = ["ball", "car", "coffee", "helmet", "teapot", "toaster"]

factors = [1]

output_dir = f"logs/{args.out}/{dataset}"
dataset_dir = f"data/{dataset}"

excluded_gpus = set([])

jobs = list(itertools.product(scenes, factors))


def run_script(command):
    exit_code = os.system(command)
    if exit_code > 0:
        print("Error %d while running %s'" % (exit_code, command), file=sys.stderr)
        sys.exit(exit_code)


def train_scene(gpu, scene, factor):
    if not args.skip_train:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene} --white_background --eval"
        print(cmd)
        run_script(cmd)

    if not args.skip_render:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene}"
        print(cmd)
        run_script(cmd)

    if not args.skip_metrics:
        cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
        print(cmd)
        run_script(cmd)

    return True


def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., releasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)

    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)