# Data Collaboration

Implementation of data collaboration : [Data Collaboration Analysis Framework Using Centralization of Individual Inetermediate Representaions for Distributed Dataset](https://ascelibrary.org/doi/full/10.1061/AJRUA6.0001058)

## Requirments
from pipfile
```
[[source]]
name = "pypi"
url = "https://pypi.org/simple"
verify_ssl = true

[dev-packages]

[packages]
numpy = "*"
matplotlib = "*"
scikit-learn = "*"
scipy = "*"
tqdm = "*"
keras = "*"
tensorflow = "*"
seaborn = "*"

[requires]
python_version = "3.7"
```

## Running

default parameters

- `num_users = 10`: number of parties
- `ndat = 100`: size of data portions
- `nanc = 2000`:  number of anchor data
- `n_neighbors = 6`: for LLE, LPP
- `d = 75`: dimension of intermediate representation
- `repeat = 5`:  number of repeat to experiment(≒epochs)
- `anc_type = 'random'`: method to create anchor data

- run default condition
```
(user)$ cd dc_dir
dc_dir (user)$ python src/dc_main.py
```
- to set non-iid
see options.py for more argments

```
dc_dir (user)$ python src/dc_main.py --iid=0
```