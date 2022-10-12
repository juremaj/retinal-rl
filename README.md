## Setting up the development environment

Unfortunately, putting together a unified build scheme has proven challenging, because our different compute resources rely on different containerization schemes (i.e. bespoke docker vs apptainer), and subtle bugs have emerged that only effect one build system or the other. We maintain a `Dockerfile` for building the docker image and the `retinal_rl.def` file to build the `apptainer` image, and we've also had success building with `conda` on bare metal.

### Apptainer

The `apptainer` image is self-contained, and building it should immediately allow running the relevant scripts in `retinal-rl`, by prefixing them with `apptainer exec [image]`. The versions of most pip packages are floating, but we have an `environment.yaml` file from a working `apptainer` build.

### Conda

Here are the steps to get a `retinal-rl` environment setup in `conda`, which should work on bare metal. First [install anaconda or miniconda](https://docs.anaconda.com/anaconda/install/index.html), and then create the environment
``` bash
conda create --name retinal-rl python=3.8 pip
conda activate retinal-rl
```
I'm using `miniconda`, so some of the following commands might be redundant if you're using `anaconda`.

Now, we use the LTS version of `pytorch` to maximize compatibility
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```
and then install `sample-factory` and `vizdoom`
```bash
pip install sample-factory
pip install vizdoom
```
Note, you may require `sample-factory=1.121.4` on a server. To avoid using an older version, a possible workaround is also to install pytorch via `conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge` and downgrade the `gym` library: `pip install gym==0.25.2`

We'll also need some other tools and libraries
```bash
conda install -c conda-forge matplotlib pyglet imageio
pip install pygifsicle
pip install openTSNE
```
IPython might also be necessary:
```bash
conda install -c conda-forge ipython
```
Finally, if you're missing some fundamental libraries, the following may help:
```bash
conda install -c conda-forge gxx boost
```

### Docker

The `Dockerfile` is a thin wrapper around the berenslab `Dockerfile` for the berenslab cluster, but may still serve as a basis for developing a `docker` container for other systems. Regardless, after building the image we then create the `conda` environment as above.

## Running retinal RL simulations

Now that we have a (hopefully) working environment, we clone the repo
```bash
https://github.com/berenslab/retinal-rl.git
```
There are three main scripts for working with `retinal-rl`:

- `train.py`: Train a model.
- `analyze.py`: Generate some analyses.
- `enjoy.py`: Watch a real time simulation of a trained agent.

Each script can be run by python in `python -m {script}`, where {script} is the name of the desired script (without the `.py` extension), followed by a number of arguments. Note that `train.py` must always be run first to create the necessary files and folders, and once run will permanently set most (all?) of the arguments of the simulation, and will ignore changes to these arguments if training is resumed.

Certain arguments must always be provided, regardless of script, namely:

- `--env`: Specifies the desired map. This will always have the form `retinal_{scenario}`, where scenario is the shared name of one of the `.wad`/`.cfg` file pairs in the `scenarios` directory.
- `--algo`: The training algorithm; for now this should always be `APPO`.
- `--experiment`: The directory under the `train_dir` directory where simulation results are saved.

The following argument should always be set when training for the first time:

- `--encoder_custom`: The options are `simple`, which is a small, hard-coded network that still tends to perform well, and `lindsey`, which has a number of tuneable hyperparameters.

For specifying the form of the `lindsey` network, the key arguments are:

- `--global_channels`: The number of channels in each CNN layers, except for the bottleneck layer.
- `--retinal_bottleneck`: Number of channels in the retinal bottleneck.
- `--vvs_depth`: Number of CNN layers in the ventral stream network.
- `--kernel_size`: Size of the kernels.

Finally, when training a model there are a number of additional parameters for controlling the reinforcement learning brain, and adjusting simulation parameters. The key ones to worry about are

- `--hidden_size`: The size of the hidden/latent state used to represent the RL problem.
- `--num_workers`: This is the number of simulation threads to run. This shouldn't be more than the number of cores on the CPU, and can be less if the simulation is GPU bottlenecked.
