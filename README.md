# Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction

It is the official repository of our work "Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction," which has been published in Transactions on Machine Learning Research (TMLR).

## Installation

Firstly, install `torch` and `torchvision` considering your machine. For example, if you want to install them in a CPU-only machine, use the following.
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Details can be found [here](https://pytorch.org/get-started/locally/).

Secondly, install `MinkowskiEngine`. If you are using a machine with GPUs, you can just install it with `pip install MinkowskiEngine`. However, if your are install it in a CPU-only machine, we should command the following.

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --cpu_only
```
Details can be found [here](https://nvidia.github.io/MinkowskiEngine/quick_start.html). If you encounter an error related to OpenBLAS, refer to [this link](https://github.com/NVIDIA/MinkowskiEngine/issues/300#issuecomment-763343048).

Then, install other dependencies following the command below.
```bash
pip install -r requirements.txt
```

## How to download the dataset
Download link will be uploaded soon.

## How to train
Run the following command to train the model to build chairs.
```bash
python main.py -c configs/0.25bce_sup_chair_pthresh0.5_150step_20eq_nskip8.yaml
```

## How to inference
Inference script will be uploaded soon.

## Citation

```
@article{AhnS2024tmlr,
    title = {Budget-Aware Sequential Brick Assembly with Efficient Constraint Satisfaction},
    author = {Ahn, Seokjun and Kim, Jungtaek and Cho, Minsu and Park, Jaesik},
    journal = {Transactions on Machine Learning Research},
    year = {2024}
}
```

## Acknowledgements
This code is based on [PyTorch implemtation of Generative Cellular Automata](https://github.com/96lives/gca).