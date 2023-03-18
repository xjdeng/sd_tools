from fastai.vision.all import *
from path import Path as path

def run(mydir, bs = 16, thresh = 0.1, cpu = False, modfile = None, place = 0):
    if not modfile:
        modfile = path(__file__).abspath().dirname() + "/model"
    gooddir = "{}/good".format(mydir)
    baddir = "{}/bad".format(mydir)
    path(gooddir).mkdir_p()
    path(baddir).mkdir_p()
    learn = load_learner(modfile, cpu=cpu)
    files = path(mydir).files()
    if len(files) == 0:
        return
    test_dl = learn.dls.test_dl(files, with_label=True)
    test_dl.bs = bs
    pred_probas, _, _ = learn.get_preds(dl=test_dl, with_decoded=True)
    prob_bad = [p[place] for p in pred_probas.numpy()]
    for f,p in zip(files, prob_bad):
        if p > thresh:
            f.move(baddir)
        else:
            f.move(gooddir)