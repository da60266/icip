from io import StringIO,BytesIO
#from PIL import Image
#import numpy as np
import matplotlib.pyplot as plt
from .edict import edict

plt.switch_backend('agg')

vis_config = edict(
    server = 'http://127.0.0.1',
    port = 31540,
    env = 'main',
)

viz=""

class set_draw(object):
    def __init__(self, server=None, port=None, env=None, **subplot_kwargs):
        #self._config = edict(vis_config)
        #vis_config.server = server if server is not None else vis_config.server
        #vis_config.port = port if port is not None else vis_config.port
        #vis_config.env = env if env is not None else vis_config.env
        self.viz = viz#visdom.Visdom(**vis_config)
        self.fig = None
        self.axes = None
        self.name = subplot_kwargs.pop('name') if 'name' in subplot_kwargs.keys() else 'default'
        if not (server or port):
            self.subplots(name=self.name, **subplot_kwargs)

    def __enter__(self):
        assert self.fig is not None and self.axes is not None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # p = Process(target=self._remote_draw)
        # p.start()
        self._remote_draw()

    def _remote_draw(self):
        strio = StringIO()
        self.fig.savefig(strio, format="svg")
        plt.close(self.fig)
        #img = Image.open(strio)
        #img = np.uint8(img)
        #img = np.transpose(img, (2,0,1))
        self.viz.svg(svgstr=strio.getvalue(), win=self.name)
        #self.viz.image(img, win=self.name)
        #vis_config.update(self._config)
        #self.viz.close()
        #self.viz = visdom.Visdom(**vis_config)

    def subplots(self, name, **plt_subplot_kwargs):
        self.name = name
        self.fig, self.axes = plt.subplots(constrained_layout=True, **plt_subplot_kwargs)
        #plt.gca().set_aspect(1)
        return self

    def close(self):
        self.viz.close(self.name, self.env)

    def __getattr__(self, item):
        return getattr(self.axes, item)
