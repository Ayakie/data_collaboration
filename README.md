# Data Collaboration

Implementation of data collaboration : [Data Collaboration Analysis Framework Using Centralization of Individual Inetermediate Representaions for Distributed Dataset](https://ascelibrary.org/doi/full/10.1061/AJRUA6.0001058)

## Requirments

- Docker desktop: version 2.3.0.3

Environment is created by docker using [poetry](https://python-poetry.org/docs/).

All requirements written in pyproject.toml are installed into container .

From pyproject.toml
```
[tool.poetry.dependencies]
python = "^3.7"
tqdm = "^4.46.0"
keras = "^2.2.4"
seaborn = "^0.10.1"
scikit-learn = "^0.23.0"
matplotlib = "^3.2.1"
scipy = "^1.4.1"
tensorflow = "^2.2.0"
numpy = "^1.18.4"
```

## Files

- dc_main.py: Increasing the number of users (`num_users`), fixing the size of data per user (`ndat`)
- dc_ndat.py: Increasing the size of data per user, fixing the number of users 
- fed_main.py & fed_ndat.py: Federated Learning ver.

## Running

default parameters

- `num_users = 5`: Number of parties
- `ndat = 100`: Size of data portions
- `nanc = 500`:  Number of anchor data
- `n_neighbors = 6`: For LLE, LPP
- `d_ir = 5`: Dimension of intermediate representation
- `repeat = 10`:  Number of repeat to experiment(≒epochs)
- `anc_type = 'random'`: Method to create anchor data
- `dataset = 'fashion_mnist'`: Dataset ['mnist', 'fashion_mnist']

Create environment
```
$ git clone https://github.com/Ayakie/data_collaboration
$ docker-compose up --build
```

Run default condition
```
$ python src/dc_main.py
```

To set non-iid and not saving figures.<br> `options.py` for more argments

```
$ python src/dc_main.py --iid=0 --save_fig=0
```