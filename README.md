### graphwave

#### Installation

```
conda create -n graphwave_env python=2.7 pip -y
source activate graphwave_env
pip install -r requirements.txt
!! Need to also install "cupy" if you want GPU support
```

#### Usage

See `./run.sh` for usage.

See original [graphwave](https://github.com/snap-stanford/graphwave) implementation for some more details about the method