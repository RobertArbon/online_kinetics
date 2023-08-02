# Online Kinetics

Online kinetics trains a deep kinetic models of biomolecular simulations (using the [`VAMPnets`](https://www.nature.com/articles/s41467-017-02388-1) 
using [`Deeptime`](https://deeptime-ml.github.io/latest/index.html) package as inspiration) in an *online* manner: 
i.e., using 1 epoch of training. 


The motivation for this idea is that unbiased simulations used to create kinetic models can be quite large and potentially
prohibitively expensive to train, especially in automated adaptive sampling methods (e.g., [Casalino et. al.](https://journals.sagepub.com/doi/full/10.1177/10943420211006452))

The intended use case is that approximate models are trained online for EDA purposes and then more accurate models 
trained in a batch / multi-epoch manner.  
Alternatively, one could use an approximate online model as a pre-processing step in creating an approximate discrete 
Markov state model (for an e.g., see [here](https://deeptime-ml.github.io/latest/notebooks/examples/ala2-example.html)) 
for use with adaptive sampling.  

The current method uses Hedged Back Propagation: [Sahoo, D. et al. (2017)](http://arxiv.org/abs/1711.03705) 
There are tentative plans to use the self-expanding neural networks of [Mitchel et. al. (2023)](https://arxiv.org/abs/2307.04526)

## Repo guide

`celerity` - a package for training (online and batch)  VAMPnet models of molecular kinetics. 
`data` - dataset are stored here. 
`docs` - WIP - the documentation files go here. 
`models` - trained and serialized models. 
`notebooks` - analysis of models and data. 
`tests` - unit and integration tests


## Installation

Assuming a linux OS with CUDA drivers 11.3 - see [Pytorch.org](pytorch.org) for details for other distros/CUDA versions.

### Conda
```
conda create -n onlinekinetics python==3.9 -y
conda activate onlinekinetics 
conda env update -f environment.yaml
pip install -qe . 
```

### Pip
Create a virtual environment your favourite way and then: 
```
pip install -r requirements.txt
pip install -qe . 
```

## Usage
