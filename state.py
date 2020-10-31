import sys,os
import mlb
import plot,test,train,fix
from util import *

class Poisoned: pass

class State:
    def __init__(self):
        self.no_pickle = []
        self.cfg = None
    # these @properties are kinda important. Putting them in __init sometimes gives weird behavior after loading
    @property
    def state(self):
        return self
    @property
    def self(self):
        return self
    @property
    def as_kwargs(self):
        kwargs = {'state': self.state}
        kwargs.update(self.__dict__)
        return kwargs
    def to(self,device):
        return
    def main_new(self,cfg):
        """
        wrapper for state.new()
        """
        print("no file to load from, creating new state...")
        if cfg.device != 'cpu':
            with torch.cuda.device(cfg.device):
                self.new(cfg=cfg)
        else:
            self.new(cfg=cfg)
        # seems to initialize on gpu anyways sometimes
        self.to(cfg.device)
    def main_load(self,cfg):
        paths = outputs_regex(cfg.load)
        # path must at least be DATE/TIME, possibly DATE/TIME/...
        paths = [p for p in paths if len(p.parts) >= 2]
        if len(paths) == 0:
            mlb.red(f'load regex `{cfg.load}` yielded no files:')
        if len(paths) != 1:
            mlb.red(f'load regex `{cfg.load}` yielded multiple possible files:')
            for path in paths:
                mlb.red(f'\t{path}')
            sys.exit(1)
        [path] = paths

        if 'saves' not in path.parts:
            savefile = 'autosave.'+str(max([int(x.name.split('.')[1])
                                    for x in (get_datetime_path(path) / 'saves').glob('autosave.*')]))
            path = get_datetime_path(path) / 'saves' / savefile
        
        assert all(['=' in arg for arg in sys.argv[1:]])
        overrides = [arg.split('=')[0] for arg in sys.argv[1:]]

        device = None
        if 'device' in overrides:
            device=cfg.device # override the device

        mlb.green(f"loading from {path}...")
        self.load(path, device=device)

        #mlb.purple(f"It looks like the device of the model is {self.phead.output[0].weight.device}")

        if cfg.mode == 'device':
            mlb.green(f"DEVICE: {self.cfg.device}")
        print("loaded")

        print_overrides = []
        for override in overrides:
            try:
                # eg override = 'data.T'
                dotpath = override.split('.')
                if dotpath[-1] == 'device':
                    raise NotImplementedError
                target = self.cfg # the old cfg
                source = cfg # the cfg that contains the overrides
                for attr in dotpath[:-1]: # all but the last one (which we'll use setattr on)
                    target = target[attr]
                    source = source[attr]
                overrided_val = source[dotpath[-1]]
                print_overrides.append(f'overriding {override} to {overrided_val}')
                with open_dict(target): # disable strict mode
                    target[dotpath[-1]] = overrided_val
            except Exception as e:
                mlb.red(e)
                pass
        state.print_overrides = '\n'.join(print_overrides)

    def new(self,cfg):
        """
        Initialize dataloaders, models, etc
        """

        self.cwd = os.getcwd()
        self.name = cfg.name

        if cfg.prefix is not None:
            self.name = cfg.prefix + '.' + self.name
        

        


        params = itertools.chain.from_iterable([head.parameters() for head in heads])
        optimizer = torch.optim.Adam(params, lr=cfg.optim.lr, eps=1e-3, amsgrad=True)

        vhead = InvalidIntermediatesValueHead(cfg)
        astar = make_solver(cfg.data.test.solver,vhead,phead,cfg.data.train.max_depth)
        j=0
        frontiers = None

        loss_window = 1
        plosses = []
        vlosses = []

        #taskloader.check()

        self.update(locals()) # do self.* = * for everything
        self.post_load()
    
    @contextlib.contextmanager
    def saveable(self):
        temp = {}
        for key in self.no_pickle:
            temp[key] = self[key]
            self[key] = Poisoned
        
        saveables = []
        for k,v in self.__dict__.items():
            if isinstance(v,State):
                continue # dont recurse on self
            if hasattr(v,'saveable'):
                saveables.append(v)
        try:
            with contextlib.ExitStack() as stack:
                for saveable in saveables:
                    stack.enter_context(saveable.saveable()) # these contexts stay open until ExitStack context ends
                yield None
        finally:
            for key in self.no_pickle:
                self[key] = temp[key]

    def save(self, locs, name):
        """
        use like state.save(locals(),"name_of_save")
        """
        self.update(locs)

        if not os.path.isdir('saves'):
            os.mkdir('saves')
        path = f'saves/{name}'
        print(f"saving state to {path}...")

        with self.saveable():
            torch.save(self, f'{path}.tmp')

        print('critical step, do not interrupt...')
        shutil.move(f'{path}.tmp',f'{path}')
        print("done")
    def load(self, path, device=None):
        if device is not None:
            device = torch.device(device)
        path = utils.to_absolute_path(path)
        print("torch.load")
        state = torch.load(path, map_location=device)
        print("self.update")
        self.update(state.__dict__)
        print("self.post_load")
        self.post_load()
        print("loaded")
    def post_load(self):
        print(f"chdir to {self.cwd}")
        os.chdir(self.cwd)
        self.init_tensorboard()
        for k,v in self.__dict__.items():
            if isinstance(v,State):
                continue # dont recurse on self
            if hasattr(v,'post_load'):
                v.post_load()
    def init_tensorboard(self):
        print("intializing tensorboard")
        self.w = SummaryWriter(
            log_dir=self.name,
            max_queue=1,
        )
        print("writer for",self.name)
        print("done")
        self.no_pickle.append('w')
    def rename(self,name):
        if self.name == name:
            return
        old_name = self.name
        self.name = name

        # shut down tensorboard since it was using the old name
        if hasattr(self,w) and self.w is not Poisoned:
            self.w.flush()
            self.w.close()
            del self.w

        # move old tensorboard files
        os.rename(old_name,self.name)
        
        self.init_tensorboard() # reboot tensorboard
    def __getitem__(self,key):
        return getattr(self,key)
    def __setitem__(self,key,val):
        return setattr(self,key,val)
    def __repr__(self):
        body = []
        for k,v in self.__dict__.items():
            body.append(f'{k}: {repr(v)}')
        body = '\n\t'.join(body)
        return f"State(\n\t{body}\n)"
    def update(self,dict):
        for k,v in dict.items():
            if hasattr(type(self), k) and isinstance(getattr(type(self),k), property):
                continue # dont overwrite properties (throws error)
            self[k] = v

