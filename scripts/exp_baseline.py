import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from psychohistory.pipeline import prepare
from psychohistory.models import Ensemble, LogitModel, GBMModel, PsychoNet, NaiveModel
from psychohistory.evaluate import evaluate, log

class Wrap:
    def __init__(self, m, groups=False): self.m, self.g = m, groups
    def fit(self, X, Y, g):
        (self.m.fit(X, Y, g) if self.g else self.m.fit(X, Y)); return self
    def predict(self, X): return self.m.predict(X)

p, base, X, Yl = prepare()
for name, f in [("baseline:naive", lambda: Wrap(NaiveModel())), ("baseline:logit", lambda: Wrap(LogitModel())),
                ("baseline:gbm", lambda: Wrap(GBMModel())), ("baseline:psychonet", lambda: Wrap(PsychoNet(members=8), True)),
                ("baseline:ensemble(avg)", lambda: Wrap(Ensemble(members=8), True))]:
    s, out, _ = evaluate(p, X, Yl, f)
    log(name, s, out)
