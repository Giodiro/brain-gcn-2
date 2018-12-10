import os, sys
import numpy as np



from sacred import Experiment



ex = Experiment("coherence_exp", ingredients=[])


@ex.config
def cfg():
    pass









@ex.automain
def main(_log, _config, _run):
    pass
