from __future__ import division

from glob import glob
import logging
import itertools
import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

from collections import defaultdict
from fnmatch import fnmatch

import forgi
import forgi.threedee.utilities.vector as ftuv
import forgi.threedee.utilities.pdb as ftup


log = logging.getLogger(__name__)

__all__ = ["DataStore", "plot_fit"]
def mean(x):
    return sum(x)/len(x)

def get_allatom_pdd(filename, stepsize):
    chains = ftup.get_all_chains(filename)[0]
    atoms = []
    for chain in chains:
        for res in chain:
            for atom in res:
                atoms.append(atom.coord)
    x,y = ftuv.pair_distance_distribution(atoms, stepsize)
    df = pd.DataFrame({"step": x, "count":y})
    return df

def area_between_curves(arr1, arr2):
    t_len = max(len(arr1), len(arr2))
    deviation = np.zeros(t_len)
    deviation[:len(arr1)]=arr1/sum(arr1)
    deviation[:len(arr2)]-=arr2/sum(arr2)
    return np.sum(np.abs(deviation))

def _energy(m, ref_kde, tar_kde):
    ref_val = ref_kde(m)
    tar_val = tar_kde(m)
    energy = (np.log( tar_val ) - np.log(ref_val))
    return -energy


def gaussian(mu, sig):
    def g_inner(x):
        return np.exp((-((x - mu)/sig)**2.)/2) / (np.sqrt(2.*np.pi)*sig)
    return g_inner

def pdd_with_errors(pdd, a):
    e = np.maximum(np.abs(pdd["error"]), 5*10**-8)
    y_low = np.maximum(pdd["count"]-a*e, 0)
    y_high = pdd["count"]+a*e
    return pdd["count"]/sum(pdd["count"]), y_low/sum(pdd["count"]), y_high/sum(pdd["count"]),

class _LoggingContext(object):
    """
    Modified from the python logging coockbook: https://docs.python.org/3/howto/logging-cookbook.html
    """
    def __init__(self, logger, level=None):
        self.logger = logger
        self.level = level
    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        # implicit return of None => don't swallow exceptions

class DataStore(object):
    def __init__(self, target_pdd_filename):
        """
        :param target_pdd_filename: A filename to read the target pair distance
                                    distribution function from. It should be a
                                    comma, separated file with 3 columns with
                                    the following headers:
                                    "distance", "count","error"
        """
        self.target_pdd_fn = target_pdd_filename
        self.ref = pd.read_csv(target_pdd_filename)
        stepsize = (self.ref["distance"].values[-1]-self.ref["distance"].values[0])/(len(self.ref)-1)
        distdiff = self.ref["distance"].values[1:]-self.ref["distance"].values[:-1]
        if np.all(np.abs(distdiff/stepsize-1)<0.1):
            self.stepsize = stepsize
        else:
            raise ValueError("Unequally spaced target distribution. Stepsizes are {}, avg {}", distdiff, stepsize)
        self.pdds={}
        self.cgs ={}
        self.deviations={}
        self.chis = {}
    def plot_ref(self, ax, a=1):
        """
        Plot the reference pair distance distribution to the provided axes
        """
        y, y_low, y_high = pdd_with_errors(self.ref, a)
        ax.plot(self.ref["distance"], y_low, color="orange")
        ax.plot(self.ref["distance"], y_high, color="orange")
        ax.plot(self.ref["distance"], y, color="red")

    def get_scores(self, pattern):
        """
        :param pattern: A globbing pattern for a filename or a list of patterns.
        """
        if not isinstance(pattern, list):
            pattern = [pattern]
        ref = self.ref
        curr_deviations={}
        for fn, pdd in self.pdds.items():
            for p in pattern:
                if fnmatch(fn, p):
                    if fn not in self.deviations:
                        deviation = area_between_curves(pdd[1], ref["count"])
                        self.deviations[fn]=deviation
                    curr_deviations[fn]=self.deviations[fn]
                    break

        return list(sorted(curr_deviations.items(), key=lambda x: x[1]))

    def score_binned(self, fn, n):
        """
        :param n: put n original bins into one bin
        """
        def binned(arr):
            pass
            binned = np.zeros(int(len(arr/n)+1))
            for i in range(n):
                a=arr[i::n]
                binned[:len(a)]+=a[:len(binned)]
            return binned
        deviation = area_between_curves(binned(self.pdds[fn][1]), binned(self.ref["count"]))
        return deviation


    def plot(self, pattern, what="all", show=True):
        scores = self.get_scores(pattern)
        if what=="all":
            indices=range(len(scores))
        elif what=="b" or what=="best":
            indices=[0]
        elif what=="w" or what=="worst":
            indices=[-1]
        elif what=="bw" or what=="best_worst":
            indices=[0,-1]
        axs = []
        for i in indices:
            fn, score=scores[i]
            fig,ax = plt.subplots(figsize=(10,8))
            axs.append(ax)
            ax.plot(self.pdds[fn][0], self.pdds[fn][1]/sum( self.pdds[fn][1]))
            plt.title("{}:{}".format(score, fn))
            self.plot_ref(ax, 50)
        if show:
            plt.show()
        return axs

    def load(self, pattern):
        num_loaded = 0
        num_selected=0
        if not isinstance(pattern, list):
            pattern = [pattern]
        fns = []
        for pat in pattern:
            fns.extend(glob(pat))
        if len(fns)>1000:
            lev=logging.WARNING
        else:
            lev = logging.INFO
        log.log(lev, "Loading %s files", len(fns))
        with _LoggingContext(logging.getLogger(), logging.CRITICAL):
            for fn in fns:
                num_selected+=1
                if fn not in self.cgs:
                    cgs = forgi.load_rna(fn)
                    if len(cgs)!=1:
                        raise ValueError("Expected 1 RNA component in file {}, found {}:{}".format(fn, len(cgs)), [cg.name for cg in cgs])
                    cg, = cgs
                    cg.infos["filename"] = fn
                    self.cgs[fn] = cg
                    num_loaded+=1
                if fn not in self.pdds:
                    points = []
                    try:
                        pd_pdd = pd.read_csv(fn+".pdd.csv")
                    except:
                        for i in range(1,cg.seq_length+1):
                            points.append(cg.get_virtual_residue(i, allow_single_stranded=True))
                        x,y = ftuv.pair_distance_distribution(points, stepsize=self.stepsize)
                        df = pd.DataFrame({"step": x, "count":y})
                        df.to_csv(fn+".pdd.csv")
                    else:
                        x=pd_pdd["step"]
                        y=pd_pdd["count"]
                    self.pdds[fn]=(x,y)
        scores = self.get_scores(pattern)
        if scores:
            minsc = scores[0]
            maxsc = scores[-1]
        else:
            minsc=None
            maxsc=None
        return num_selected, num_loaded, minsc, maxsc


    def plot_target_distr(self, pattern, axes=None, snapshots=(25,50,75,110), show=True, a=50):
        target_values, lower_val, upper_val = pdd_with_errors(self.ref, a)
        if not isinstance(pattern, list):
            pattern = [pattern]
        values = defaultdict(list)
        ref = defaultdict(list)
        positions = {}

        if axes is None:
            fig,ax = plt.subplots(figsize=(10,8))
        else:
            ax = axes[0]
        self.plot_ref(ax, a)
        for fn, pdd in self.pdds.items():
            for p in pattern:
                if fnmatch(fn, p):
                    normed_pdd = pdd[1]/sum(pdd[1])
                    label = os.path.splitext(os.path.basename(fn))[0]
                    ax.scatter(pdd[0], normed_pdd, label=label, s=10)
                    for i in snapshots:
                        try:
                            values[i].append(normed_pdd[i])
                        except Exception:
                            pass
                        else:
                            positions[i]=pdd[0][i]
                    break
        y, y_low, y_high = pdd_with_errors(self.ref, a)
        ax.set_ylim([0,max(y)*1.1])

        for i,key in enumerate(snapshots):
            if key not in values:
                continue
            if axes is None:
                fig,ax = plt.subplots(figsize=(10,8))
            else:
                ax=axes[i+1]
            ax.set_title(positions[key])
            span= upper_val[key]-lower_val[key]
            x = np.linspace(max(0,lower_val[key]-span/2), upper_val[key]+span, 300)
            gaus = gaussian(target_values[key], target_values[key]-lower_val[key])
            ax.plot(x, gaus(x), label="target", color="green")
            try:
                kde = scipy.stats.gaussian_kde(values[key])
            except scipy.linalg.LinAlgError:
                log.exception("Cannot plot reference distribution: ")
            else:
                ax.plot(x, kde(x), label="reference")
                ax2 = ax.twinx()
                x2 = np.linspace(max(0,lower_val[key]-0.003), upper_val[key]+0.003, 200)
                ax2.plot(x2, _energy(x2, kde, gaus), color="red", label="Energy")
                ax2.legend(loc="upper right")
            true_val = target_values[key]

            maxy = gaus(true_val)
            ax.plot([true_val, true_val], [0,maxy],color="red" )
            ax.plot([lower_val[key], lower_val[key]], [0,maxy],color="orange" )
            ax.plot([upper_val[key], upper_val[key]], [0,maxy],color="pink" )
            ax.legend(loc="upper left")
            #plt.savefig("rrm_{}.svg".format(key))
        if show:
            plt.show()

    def get_chi(self, fn, suffix="00.fit"):
        if fn not in self.chis:
            with open(os.path.splitext(fn)[0]+suffix) as f:
                line=next(f)
            fields=line.split("Chi^2:")
            chi=float(line.split("Chi^2:")[1].strip())
            self.chis[fn]=chi
        return self.chis[fn]


def plot_fit(fn, ax=None, show=True, square=False):
    data = pd.read_csv(fn,
                       delim_whitespace=True, skiprows=1, names=["c0", "c1", "c2", "c3"])
    if ax is None:
        fig,ax = plt.subplots()
    data.c1=np.maximum(data.c1,0)
    data.c3=np.maximum(data.c3,0)
    if square:
        x_vals=data.c0**2
        ax.set_xlabel("q^2")
    else:
        x_vals=data.c0
        ax.set_xlabel("q")
    ax.scatter(x_vals, np.log(data.c1), label="experiment")
    ax.plot(x_vals, np.log(data.c3), color="red", label="fit")
    ax.set_ylabel("log(I)")
    ax.legend()
