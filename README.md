# Data Collaboration

Implementation of data collaboration : [Data Collaboration Analysis Framework Using Centralization of Individual Inetermediate Representaions for Distributed Dataset](https://ascelibrary.org/doi/full/10.1061/AJRUA6.0001058)

## Requirments

Environment is created by [Pipenv](https://pipenv-ja.readthedocs.io/ja/translate-ja/basics.html)

From Pipfile

```
[[source]]
name = "pypi"
url = "https://pypi.org/simple"
verify_ssl = true

[packages]
numpy = "==1.18"
scikit-learn = "==0.22.2"
keras = "==2.3.1"
tensorflow = "==2.1.0"
scipy = "==1.4.1"
matplotlib = "==3.2.1"
seaborn = "==0.10.0"
tqdm = "*"
pandas = "==1.0.3"

[requires]
python_version = "3.7"
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


Run default condition
```
(user)$ cd dc_dir
dc_dir (user)$ python src/dc_main.py
```

To set non-iid<br> `options.py` for more argments

```
dc_dir (user)$ python src/dc_main.py --iid=0
```