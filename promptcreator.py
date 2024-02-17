from path import Path as path
try:
    import imagetools
except ImportError:
    from . import imagetools

from abc import ABC, abstractmethod
import json
import sys

class Action(ABC):

    @abstractmethod
    def execute(texts):
        pass

class Replace(Action):

    def __init__(self, a, b = None):
        if (b is not None) and (len(b) > 0):
            self.replacements = [(a, b)]
        else:
            self.replacements = a
    
    def execute(self, texts):
        if len(self.replacements) == 1:
            a,b = self.replacements[0]
            return [text.replace(a,b) for text in texts]
        else:
            return [imagetools.replace_substrings(text, self.replacements) for text in texts]

class Filter(Action):

    def __init__(self, pos = [], neg = []):
        if isinstance(pos, str):
            pos = [pos]
        if isinstance(neg, str):
            neg = [neg]
        self.pos = pos
        self.neg = neg
    
    def execute(self, texts):
        for pos in self.pos:
            if isinstance(pos, str):
                texts = [text for text in texts if pos in text]
            else:
                newtexts = []
                for text in texts:
                    add = False
                    for p in pos:
                        if p in text:
                            add = True
                    if add:
                        newtexts.append(text)
                texts = newtexts
        for neg in self.neg:
            if isinstance(neg, str):
                texts = [text for text in texts if neg not in text]
            else:
                newtexts = []
                for text in texts:
                    add = False
                    for ng in neg:
                        if ng not in text:
                            add = True
                    if add:
                        newtexts.append(text)
                texts = newtexts
        return texts

class Project:

    def __init__(self, x, recursive = True):
        self.texts = []
        p = path(x)
        if p.isdir():
            if recursive:
                files = list(p.walkfiles())
            else:
                files = p.files()
            textfiles = [f for f in files if f.ext.lower() == ".txt"]
            for tf in textfiles:
                with open(tf, 'r') as f:
                    self.texts += f.read().split("\n")
        else:
            with open(p, 'r') as f:
                self.texts += f.read().split("\n")

    def execute(self, actions):
        texts = self.texts
        for action in actions:
            texts = action.execute(texts)

        return texts
    
    def execute_json(self, jsonfile, outputfile = "output.txt"):
        parsed = json.load(open(jsonfile))
        actions = []
        for p in parsed:
            mclass = Action
            if p['type'].lower() == "filter":
                mclass = Filter
            elif p['type'].lower() == "replace":
                mclass = Replace
            actions.append(mclass(p['a'], p['b']))
        output = self.execute(actions)
        with open(outputfile,'w') as f:
            f.write("\n".join(output))

def execute(inputfile, jsonfile, outputfile = "output.txt"):
    p = Project(inputfile)
    p.execute_json(jsonfile, outputfile)

if __name__ == "__main__":
    inputfile = sys.argv[1]
    jsonfile = sys.argv[2]
    if len(sys.argv) > 3:
        outputfile = sys.argv[3]
    else:
        outputfile = "output.txt"
    execute(inputfile, jsonfile, outputfile)