from util import *
unthread() # turn off numpy and torch threading

import mlb
import plot,test,train,fix
import profile,cmd
from state import State

import sys,os
import contextlib
import numpy as np
import torch

def create_state(cfg):
    state = State()
    if cfg.load is None:
        state.main_new()
    else:
        state.main_load()
    which(state.cfg)
    deterministic(state.cfg.seed)
    mlb.purple(state.print_overrides)
    return state

@hydra.main(config_path="conf", config_name='config')
def hydra_main(cfg):
    # if cfg.debug.verbose:
    #     mlb.set_verbose()

    np.seterr(all='raise') # so we actually get errors when overflows and zero divisions happen

    def on_crash():
        print(os.getcwd())
         
    with mlb.debug(debug=cfg.debug.mlb_debug, ctrlc=on_crash, crash=on_crash):
        with (torch.cuda.device(state.cfg.device) if state.cfg.device != 'cpu' else contextlib.nullcontext()):
            # PRINT
            if cfg.print:
                if cfg.load:
                    state = create_state(cfg)
                    which(state.cfg)
                else:
                    which(cfg)
                print("cfg.print was specified, exiting")

            # PLOT
            elif cfg.mode == 'plot':
                plot.main(cfg)

            # TEST
            elif cfg.mode == 'test':
                test.main_pre(cfg)
                state = create_state(cfg)
                mlb.yellow("===START===")
                test.main(cfg, state)
            
            # CMD
            elif cfg.mode == 'cmd':
                cmd.main(cfg)

            # TRAIN
            elif cfg.mode == 'resume':
                mlb.yellow("===START===")
                print("Entering training loop...")
                train.main(**state.as_kwargs)

            # PROFILE
            elif cfg.mode == 'profile':
                profile.main(cfg)

            # INSPECT
            elif cfg.mode == 'inspect':
                print()
                print("=== Inspecting State ===")
                which(state.cfg)
                breakpoint()
                raise Exception("take a look around")

            else:
                mlb.die(f"Mode not recognized: {cfg.mode}")
        mlb.yellow("===END===")

if __name__ == '__main__':
    hydra_main()
