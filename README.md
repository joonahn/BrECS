# Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction

It is the official repository of our work "Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction," which has been published in Transactions on Machine Learning Research (TMLR).

## Installation

Firstly, install `torch` and `torchvision` considering your machine. For example, if you want to install them in a CPU-only machine, use the following.
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Details can be found [here](https://pytorch.org/get-started/locally/).

Secondly, install `MinkowskiEngine`. If you are using a machine with GPUs, you can just install it with `pip install MinkowskiEngine`. However, if your are install it in a CPU-only machine, we should command the following.

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --cpu_only
```
Details can be found [here](https://nvidia.github.io/MinkowskiEngine/quick_start.html).

Then, install other dependencies following the command below.
```
pip install -r requirements.txt
```

## Citation

```
@article{AhnS2024tmlr,
    title = {Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction},
    author = {Ahn, Seokjun and Kim, Jungtaek and Cho, Minsu and Park, Jaesik},
    journal = {Transactions on Machine Learning Research},
    year = {2024}
}
```
