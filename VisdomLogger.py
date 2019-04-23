import visdom
import markdown
import numpy as np
import time

class VisdomLogger(object):
    
    def __init__(self, environment_name=time.strftime("%d/%m/%Y/%H:%M:%S"), checkpoint_interval=10):
        self.vis = visdom.Visdom(env=environment_name)
        self.env_name = environment_name
        self.win_opts = dict()
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_n_logs_ago = 0
        
    def checkpoint(self, now=False):
        self.checkpoint_n_logs_ago += 1
        if now:
            self.checkpoint_visualization()
            self.checkpoint_n_logs_ago = 0
        elif self.checkpoint_n_logs_ago % self.checkpoint_interval == 0:
            self.checkpoint_visualization()
            self.checkpoint_n_logs_ago = 0
    
    def checkpoint_visualization(self):
        """ keep results even if the server crashes """
        self.vis.save(envs=[self.env_name]) 
        
    def add_markdown(win_name='Markdown', markdown_string=r'#A markdown string', html_format='html5'):
        self.vis.text(markdown.markdown(markdown_string, output_format='html5'), win=win_name)
        self.checkpoint_visualization()
        
    def create_plot_window(self, x_label, y_label, win_name, title=None, trace_name=None):
        if title is None:  # best to make 
            title = trace_name
        opts=dict(xlabel=x_label, ylabel=y_label, title=title, showlegend=False)
        self.win_opts[win_name] = opts  # win_name = win_id 
        self.vis.line(X=np.array([1]),
                      Y=np.array([np.nan]),
                      update=None,
                      name=trace_name,
                      win=win_name,
                      opts=opts)
    
    def add_line(self, x, y, trace_name, win_name, opts=dict(xlabel='x', ylabel='y', title='title', showlegend=True)):
        if win_name not in self.win_opts:
            update=None
            self.win_opts[win_name] = opts  # store window options once *
        else:
            update='append'
        self.vis.line(X=np.array(x),
                      Y=np.array(y),
                      name=trace_name,
                      update=update,
                      win=win_name,
                      opts=self.win_opts[win_name]) # * recall window options every time
        self.checkpoint()  # persist visualization every self.checkpoint_interval loged data pieces
        
    def close(self, win_name):
        if win_name in self.win_opts:
            self.vis.close(win_name)

