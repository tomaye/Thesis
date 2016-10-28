import numpy as np
from sklearn.model_selection import StratifiedKFold

X = ["a", "b", "c", "d"]
y = [1, 1, 2, 2]
skf = StratifiedKFold(n_splits=2)
#for train, test in kf.split(X):
#    print("%s %s" % (train, test))

splits = skf.get_n_splits(X,y)

print(splits)