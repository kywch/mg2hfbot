# mg2hfbot

Want to try LeRobot on complex tasks without a physical robot arm? 

MimicGen provides over 26,000 trajectories across 12 different tasks. `mg2hfbot` converts MimicGen datasets, trains state-of-the-art LeRobot policies (including ACT and diffusion) on these datasets, and evaluates the trained policies.

Thank you to Ajay Mandlekar and Remi Cadene for guidance, and [Puffer AI](https://github.com/PufferAI/PufferLib) for computing support.

## Getting Started

1. Clone the repository.
    ```
    $ git clone https://github.com/kywch/mg2hfbot.git
    $ cd mg2hfbot
    ```

2. Setup the virtual environment and install dependencies.

    ```
    $ python3.10 -m venv .venv
    $ source .venv/bin/activate
    (.venv) $ pip install -e .
    ```

3. Evaluate a pretrained policy.
   
   The following command downloads the pretrained policy from HuggingFace Hub and evaluates it.
   ```
   (.venv) $ python eval.py -p kywch/act_mimicgen_stack_d0
   ```

   You can also evaluate a pretrained policy from a local directory using the same `-p <directory>`. You can find the pretrained policy in the `outputs/train/<date>/<time_policy>/checkpoints/<steps>/pretrained_model`.

4. Train a policy.
    The following command trains an ACT policy on the stack_d0 task.
    ```
    (.venv) $ python train.py env=stack_d0 policy=act_mimicgen dataset_repo_id=kywch/mimicgen_stack_d0
    ```
    `stack_d0`, `act_mimicgen` and `kywch/mimicgen_stack_d0` can be replaced with other task, policy and dataset repo IDs.

5. Convert MimicGen datasets into LeRobot format.
    The following command converts the MimicGen stack_d0 dataset to LeRobot format.
    ```
    (.venv) $ python convert_to_lerobot.py --dataset_type core --task stack_d0
    ```
    `core` and `stack_d0` can be replaced with other dataset type and task. 


### Troubleshooting
* This repo assumes that you have a GPU with CUDA 12.1+ installed. TO check if your CUDA version, run:
    ```
    $ nvidia-smi
    ```

* If you encounter errors during `git clone` or `pip install -e .`, run the following command. This will fail if you don't have sudo privileges.
    ```
    $ sudo apt-get update && apt-get install build-essential cmake \
        libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg
    ```

* If you encounter errors during `python3.10 -m venv .venv`, run the following command. This will fail if you don't have sudo privileges.
    ```
    $ sudo apt-get install python3.10-dev python3.10-venv
    ```

* If you don't have a sudo privilege but can use docker, you can set up a docker container with the following command:
    ```
    $ docker run -it \
        --name pixi \
        --gpus all \
        --shm-size 8gb \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /mnt/wslg:/mnt/wslg \
        -v "$(pwd):/host" \
        -e DISPLAY \
        -e WAYLAND_DISPLAY \
        -e XDG_RUNTIME_DIR \
        -e PULSE_SERVER \
        ghcr.io/prefix-dev/pixi:jammy-cuda-12.2.2 bash
    ```
    Then, inside the docker container, run the following commands:
    ```
    # apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
        python3.10-dev python3.10-venv
    ```
* EXPERIMENTAL: The above docker container comes with `pixi` installed, which is an alternative to venv and pip. To install dependencies, run the following command in the `mg2hfbot` directory after running the above `apt-get install ...`:
    ```
    $ pixi install
    ```

## Known Issues
* The conversion and evaluation runs on a single environment, so it might take some time. I tried the vectorized environment but could not make it work.
* While converting the MimicGen datasets, not all demonstrations were successfully reproduced. On the HuggingFace Hub, the dataset containing only successful demonstrations is available with the _so suffix, like `kywch/mimicgen_stack_d0_so`.


## MimicGen datasets
MimicGen contains two types of datasets: source human demonstrations and core task variations. The convert script downloads the data into `mg_download` directory. For more details about MimicGen, see their [project website](https://mimicgen.github.io/) and the [paper](https://arxiv.org/pdf/2310.17596).

### Source Datasets 
`--dataset_type source` and `--task <task_name>` are used to download and convert the source human demonstrations. Each dataset has 10 demonstrations. 

* hammer_cleanup
* kitchen
* coffee
* coffee_preparation
* nut_assembly
* mug_cleanup
* pick_place
* square
* stack
* stack_three
* threading
* three_piece_assembly

### Core Datasets
`--dataset_type core` and `--task <task_name>` are used to download and convert the Mimicgen-generated 1000 demonstrations. d0, d1, and d2 indicates increasing difficulty.

* hammer_cleanup_d0, hammer_cleanup_d1
* kitchen_d0, kitchen_d1
* coffee_d0, coffee_d1, coffee_d2
* coffee_preparation_d0, coffee_preparation_d1
* nut_assembly_d0
* mug_cleanup_d0, mug_cleanup_d1
* pick_place_d0
* square_d0, square_d1, square_d2
* stack_d0, stack_d1
* stack_three_d0, stack_three_d1
* threading_d0, threading_d1, threading_d2
* three_piece_assembly_d0, three_piece_assembly_d1, three_piece_assembly_d2

## Trying new policies that are not included in the LeRobot repo
This repo includes the binding to RoboMimic's `bc-rnn` policy. See the `policies/robomimic_bc` directory for more details.


## Acknowledgements
* [LeRobot](https://github.com/huggingface/lerobot): Making AI for Robotics more accessible with end-to-end learning
* [MimicGen](https://mimicgen.github.io/): A Data Generation System for Scalable Robot Learning using Human Demonstrations
* [RoboMimic](https://github.com/ARISE-Initiative/robomimic): A Modular Framework for Robot Learning from Demonstration
