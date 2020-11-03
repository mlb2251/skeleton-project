import sys,os
import mlb
import plot,test,train,fix
from util import *

import time
import torch
import numpy as np
from tqdm import tqdm

def window_avg(window):
    window = list(filter(lambda x: x is not None, window))
    return sum(window)/len(window)

def main(
    state,
    cfg,
    trainloader,
    testloader,
    validloader,
    model,
    w,
    optimizer,
    loss_window,
    loss_fn,
    losses=None,
    frontiers=None,
    best_validation_loss=np.inf,
    j=0,
    **kwargs,
        ):
    print(f"j:{j}")
    """
    Run any assertion-based tests here like model.run_tests()
    """

    frontiers = [] if frontiers is None else frontiers
    losses = [] if losses is None else losses
    time_since_print = None

    while True:

        for data in tqdm(trainloader, total=len(trainloader.dataset)/trainloader.batch_size, ncols=80):
            # abort if reached end
            if cfg.loop.max_steps and j > cfg.loop.max_steps:
                mlb.purple(f'Exiting because reached maximum step for the above run (step: {j}, max: {cfg.loop.max_steps})')
                return

            # data
            inputs, labels = data
            inputs.to(cfg.device)
            labels.to(cfg.device)

            model.train()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            # for printing later
            if cfg.loop.print_every is not None: # dont gather these if we'll never empty it by printing
                losses.append(loss.item())

            mlb.freezer('pause')

            if mlb.predicate('return'):
                return
            if mlb.predicate('which'):
                which(cfg)
            if mlb.predicate('rename'):
                name = input('Enter new name:')
                state.rename(name)
                # VERY important to do this:
                w = state.w
                name = state.name

            # printing and logging
            if j % cfg.loop.print_every == 0:
                rate = len(plosses)/(time.time()-time_since_print) if time_since_print is not None else None
                if rate is None: rate = 0
                time_str = f" ({rate:.2f} steps/sec)"
                vloss_avg = sum(vlosses) / max([len(vlosses),1])
                ploss_avg = sum(plosses) / max([len(plosses),1])
                for head,loss in zip([vhead,phead],[vloss_avg,ploss_avg]): # important that the right things zip together (both lists ordered same way)
                    print(f"[{j}]{time_str} {head.__class__.__name__} {loss}")
                    w.add_scalar('TrainLoss/'+head.__class__.__name__, loss, j)
                print()
                w.flush()
                vlosses = []
                plosses = []
                time_since_print = time.time()

            # validation loss
            if cfg.loop.valid_every is not None and j % cfg.loop.valid_every == 0:
                # get valid loss
                for head in heads:
                    head.eval()
                with torch.no_grad():
                    vloss = ploss = 0
                    for f in validation_frontiers:
                        vloss += vhead.valueLossFromFrontier(f, g)
                        ploss += phead.policyLossFromFrontier(f, g)
                        if ploss.item() == np.inf:
                            breakpoint()
                    vloss /= len(validation_frontiers)
                    ploss /= len(validation_frontiers)
                # print valid loss
                for head, loss in zip([vhead,phead],[vloss,ploss]):
                    mlb.blue(f"Validation Loss [{j}] {head.__class__.__name__} {loss.item()}")
                    w.add_scalar('ValidationLoss/'+head.__class__.__name__, loss.item(), j)
                # save model if new record for lowest validation loss
                val_loss = (vloss+ploss).item()
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    state.save(locals(),'best_validation')
                    mlb.green('new lowest validation loss!')
                    w.add_scalar('ValidationLossBest/'+head.__class__.__name__, loss, j)

            # search on validation set
            if cfg.loop.search_valid_every is not None and j % cfg.loop.search_valid_every == 0:
                model_results = test.test_models([astar],
                                            validation_frontiers[: cfg.loop.search_valid_num_tasks],
                                            g,
                                            timeout=cfg.loop.search_valid_timeout,
                                            verbose=True)
                accuracy = len(model_results[0].search_results) / len(validation_frontiers[:cfg.loop.search_valid_num_tasks]) * 100
                w.add_scalar('ValidationAccuracy/'+head.__class__.__name__, accuracy, j)
                plot.plot_model_results(model_results, file='validation', w=w, j=j, tb_name=f'ValdiationAccuracy')

            # if mlb.predicate('test'): # NOT REALLY USED
            #     model_results = test_models([astar], test_tasks, g, timeout=cfg.loop.search_valid_timeout, verbose=True)
            #     plot_model_results(model_results, file='test', salt=j)

            j += 1 # increment before saving so we resume on the next iteration
            if cfg.loop.save_every is not None and (j-1) % cfg.loop.save_every == 0: # the j-1 is important for not accidentally repeating a step
                state.save(locals(),f'autosave.{j}')
