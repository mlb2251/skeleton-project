import sys,os
import mlb
import plot,test,train,fix

import hydra
from omegaconf import DictConfig,OmegaConf,open_dict
from datetime import datetime
import pathlib
from pathlib import Path


def which(cfg):
    print(yaml(cfg))
    print(getcwd())
    regex = getcwd().parent.name + '%2F' + getcwd().name
    print(f'http://localhost:6696/#scalars&regexInput={regex}')
    print("curr time:",timestamp())

def getcwd():
    return Path(os.getcwd())

def yaml(cfg):
    return OmegaConf.to_yaml(cfg)

def timestamp():
    return datetime.now()

## PATHS

def toplevel_path(p):
    """
    In:  plots/x.png
    Out: /scratch/mlbowers/proj/example_project/plots/x.png
    """
    return Path(hydra.utils.to_absolute_path(p))

def outputs_path(p):
    """
    In:  plots/x.png
    Out: /scratch/mlbowers/proj/example_project/outputs/12-31-20/12-23-23/plots/x.png
    """
    return toplevel_path('outputs') / p

def outputs_relpath(p):
    """
    In:  plots/x.png
    Out: 12-31-20/12-23-23/plots/x.png
    """
    return outputs_path(p).relative_to(outputs_path(''))

def get_datetime_path(p):
    """
    Path -> Path
    In:  .../2020-09-14/23-31-49/t3_reverse.no_ablations_first
    Out: .../2020-09-14/23-31-49
    Harmless on shorter paths
    """
    idx = p.parts.index('outputs')+3 # points one beyond TIME dir
    return pathlib.Path(*p.parts[:idx]) # only .../DATE/TIME dir

def get_datetime_paths(paths):
    return [get_datetime_path(p) for p in paths]

def outputs_regex(*rs):
    """
    The union of one or more regexes over the outputs/ directory.
    Returns a list of results (pathlib.Path objects)
    """
    res = []
    for r in rs:
        r = r.strip()
        if r == '':
            continue # use "*" instead for this case please. I want to filter out '' bc its easy to accidentally include it in a generated list of regexes
        try:
            r = f'**/*{r}'
            res.extend(list(outputs_path('').glob(r)))
        except ValueError as e:
            print(e)
            return []
    return sorted(res)


def filter_paths(paths, predicate):
    return [p for p in paths if predicate(p)]

    # then filter using predicates
    for predicate in [arg for arg in args if '=' in arg]:
        lhs,rhs = predicate.split('=')
        # TODO WAIT FIRST JUST FOLLOW THIS https://github.com/tensorflow/tensorboard/issues/785
        # idk it might be better.
        # TODO first navigate to the actual folder that the tb files are in bc thats 
        # what process() should take as input (e.g. 'tb' or whatever prefix+name is)
        process(result)
        raise NotImplementedError

    return results

def unthread():
    """
    disables parallelization
    """
    import os
    assert 'numpy' not in sys.modules, "you should call this function before importing numpy"
    assert 'torch' not in sys.modules, "you should call this function before importing torch"
    os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
    torch.set_num_threads(1)

def deterministic(seed):
    torch.manual_seed(seed)
    # warning: these may slow down your model
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)