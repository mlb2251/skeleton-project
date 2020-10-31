import sys,os
import mlb
import plot,test,train,fix
from util import *
import torch


def main_pre(cfg):
    original_cfg = None
    tests_from = cfg.test.from_fn or cfg.test.from_file # use fancy python `or` semantics
    if cfg.test.from_fn is not None:
        if cfg.test.from_fn not in test.tests.tests:
            mlb.red(f"from_fn value not recognized. options are: {list(tests.tests.keys())}")
            sys.exit(1)
        test_frontiers = test.tests.tests[cfg.test.from_fn](cfg)
        mlb.purple(f"got {len(test_frontiers)} test frontiers from {cfg.test.from_fn}()")
        if cfg.test.to_file is not None:
            print(f"Writing saved tests to {cfg.test.to_file}...")
            torch.save((test_frontiers,cfg), test.tests.tests_dir / cfg.test.to_file)
            sys.exit(0)
    elif cfg.test.from_file is not None:
        (test_frontiers,original_cfg) = torch.load(test.tests.tests_dir / cfg.test.from_file)
        # note that original_cfg is just around in case you ever want a record of how the tests were created!
        mlb.green(yaml(original_cfg))
        tests_from = cfg.test.from_file
        test_frontiers = preprocess(test_frontiers,original_cfg)
        mlb.purple(f"loaded {len(test_frontiers)} test frontiers from {cfg.test.from_file} (details in `original_cfg`)")
    else:
        mlb.die("Specify either test.from_file or test.from_fn")
    assert isinstance(test_frontiers,list) and len(test_frontiers) > 0
    if cfg.load is None:
        mlb.die("no state specified to load, exiting")

def main(cfg, state):
    if cfg.test.model_result_path is None:
        mlb.red("please specify test.model_result_path")
        sys.exit(1)
    ### NOTE: this continues from the earlier 'test' section
    if original_cfg is not None and original_cfg.test.from_fn == 'deepcoder':
        test.cfg_diff(state.cfg.data.train,original_cfg.data.test) # print the differences
    if cfg.test.max_tasks is not None and len(test_frontiers) > cfg.test.max_tasks:
        mlb.red(f"Cutting down test frontiers from {len(test_frontiers)} to {cfg.test.max_tasks}")
        test_frontiers = test_frontiers[:cfg.test.max_tasks]


    state = fix.fix_state_testmode(state)
    vhead = InvalidIntermediatesValueHead(cfg) if cfg.test.validator_vhead else SampleDummyValueHead()
    solver = make_solver(cfg.data.test.solver,vhead,state.phead,cfg.data.test.max_depth)
    if original_cfg is not None:
        if state.cfg.data.train.V != original_cfg.data.test.V:
            mlb.die(mlb.mk_bold(f"HUGE WARNING: You have trained on {state.cfg.data.train.V} data but are testing on {original_cfg.data.test.V}"))
    mlb.purple("Running tests")

    if cfg.test.scaffold:
        mlb.red(mlb.mk_bold("WARNING: SCAFFOLDING IS TURNED ON"))
        assert hasattr(test_frontiers[0],'scaffold'), "turn off scaffolding or remake ur data"
        assert test_frontiers[0].scaffold is not None, "make sure expressive lambdas are turned on"


    model_results = test.test_models([solver],
                                test_frontiers,
                                state.g,
                                timeout=cfg.test.timeout,
                                verbose=True,
                                scaffold=cfg.test.scaffold,
                                )
    mlb.purple("plotting results")
    plot.plot_model_results(model_results,
                            model_result_path=cfg.test.model_result_path,
                            file=f'{tests_from}_{cfg.test.timeout}s'
                            )


def test_models(astars, test_tasks, g, timeout, verbose=True, scaffold=False):
    """
    `astars`: a list of one or more Astar objects
        These can be easily made with makeDeepcoderData.make_solver('astar',vhead,phead,maxDepth)
    `test_tasks`: a list of Tasks or FakeFrontiers to run search on
    `g`: Grammar passed to Astar.infer()
    `timeout`: the search timeout
    """
    test_scaffolds = None
    if len(test_tasks) > 0 and isinstance(test_tasks[0], FakeFrontier):
        if scaffold:
            test_scaffolds = [f.scaffold for f in test_tasks]
        test_tasks = [f.task for f in test_tasks]

    model_results = []
    for astar in astars:
        astar.owner.policyHead.eval()
        astar.owner.valueHead.eval()
        #name = f"{astar.owner.policyHead.__class__.__name__}_&&_{astar.owner.valueHead.__class__.__name__}"
        name = astar.owner.policyHead.cfg.name
        prefix = astar.owner.policyHead.cfg.prefix
        print(f"Testing: {name}")
        search_results = []
        search_failures = []
        likelihoodModel = AllOrNothingLikelihoodModel(timeout=0.01)
        for i,task in enumerate(test_tasks):
            if scaffold:
                starting_nodes = [test_scaffolds[i]]
            else:
                starting_nodes = None
            with torch.no_grad():
                fs, times, num_progs, solns = astar.infer(
                        g, 
                        [task],
                        likelihoodModel, 
                        timeout=timeout,
                        elapsedTime=0,
                        evaluationTimeout=0.01,
                        maximumFrontiers={task: 2},
                        CPUs=1,
                        starting_nodes=starting_nodes,
                    ) 
            solns = solns[task]
            times = times[task]
            if len(solns) > 0:
                assert len(solns) == 1 # i think this is true, I want it to be true lol
                soln = solns[0]
                search_results.append(soln)
                if verbose:
                    mlb.green(f"[{i+1}/{len(test_tasks)}] solved {task.name} with {len(solns)} solns in {times:.2f}s (searched {num_progs} programs)")
                    t,d,s = get_depth(solns[0].program)
                    print(f"\t-> [T{t}d{d}s{s}] {solns[0].program}")
            else:
                if verbose: mlb.red(f"[{i+1}/{len(test_tasks)}] failed to solve {task.name} (searched {num_progs} programs)")
                search_failures.append(num_progs)
        model_results.append(plot.ModelResult(prefix=prefix,name=name, cfg=astar.owner.policyHead.cfg, search_results=search_results, search_failures=search_failures, timeout=timeout))
        if verbose: mlb.blue(f'solved {len(search_results)}/{len(test_tasks)} tasks ({len(search_results)/len(test_tasks)*100:.1f}%)\n')
    return model_results

class Tests:
    def __init__(self):
        self.tests = {}
        self.tests_dir = toplevel_path('list_tests/')
    def test(self,fn):
        self.tests[fn.__name__] = fn
tests = Tests()

@tests.test
def deepcoder(cfg):
    test_cfg = cfg.data.test
    taskloader = DeepcoderTaskloader(
        cfg=cfg,
        mode='test',
        )
    tasks = taskloader.getTasks()
    if cfg.data.test.num_templates is not None:
        assert len(tasks) == cfg.data.test.num_templates
    return tasks

def joshTasks(w):
    """
    From https://github.com/ellisk42/ec/blob/Josh/dreamcoder/domains/list/makeListTasks.py
    """
    ts = []
    import json
    if w == "1":
        directory = "data/wave1"
    elif w == "2":
        directory = "data/wave2"
    elif w == "3":
        directory = "data/wave3/json"
    elif w == "3.1":
        directory = "data/wave3.1/json"
    elif w == "final":
        directory = "data/final_wave"
    else:
        assert False
    directory = toplevel_path(directory)
    for fn in os.listdir(directory):
        if not fn.endswith(".json"):continue

        if w == "final":
            if not (fn.endswith("_1.json")):
                continue

        with open(f"{directory}/{fn}") as handle:
            data = json.load(handle)

            ts.append(Task(data.get("name",fn.split(".")[0][1:]),
                           arrow(tlist(tint),tlist(tint)),
                           [((e["i"],),e["o"])
                            for e in data["data"] ]))
    return list(sorted(ts,key=lambda t: t.name))

@tests.test
def josh(cfg):
    tasks = joshTasks(str(cfg.test.josh.wave))
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers

@tests.test
def lucas(cfg):
    from dreamcoder.domains.list.main import retrieveJSONTasks, sortBootstrap, make_list_bootstrap_tasks
    def get_tasks(f):
        return retrieveJSONTasks(utils.to_absolute_path(f))
    if cfg.test.lucas.version == 1:
        tasks = get_tasks("data/list_tasks2.json")[:105]
    elif cfg.test.lucas.version == 2:
        tasks = get_tasks("data/list_tasks2.json")[:4928]
    elif cfg.test.lucas.version == 3:
        tasks = get_tasks("data/list_tasks2.json")
    elif cfg.test.lucas.version == 'old':
        tasks = get_tasks("data/list_tasks.json") + sortBootstrap()
    elif cfg.test.lucas.version == 'boopstrap':
        tasks = make_list_bootstrap_tasks()
    else:
        raise ValueError
    frontiers = [FakeFrontier(None,task) for task in tasks]
    return frontiers

# def analyze_tasks(tasks):
#     requests = defaultdict(int)
#     for task in tasks:
#         task.request

def cfg_diff(train_cfg,test_cfg):
    mlb.magenta("Differences between train and test:")
    for key in set(test_cfg.keys()) | set(train_cfg.keys()):
        if key in ['threaded', 'num_templates', 'valid_frac', 'buf_size', 'repeat', 'print_data']:
            continue #ignore these
        if key not in test_cfg:
            mlb.yellow(f"warn: key not in test data config: {key}")
            continue
        elif key not in train_cfg:
            mlb.yellow(f"warn: key not in train data config: {key}")
            continue
        if test_cfg[key] != train_cfg[key]:
            mlb.magenta(mlb.mk_bold(f"\t{key=} {train_cfg[key]=} {test_cfg[key]=}"))

