# %%
# demo of PCA for the LinQ SEAL project
# author: @aditya-sengupta
# date: 2026-11-18

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
# %%
# I'm going to write an actual implementation of the thing we drew on the board last week
xdata = np.array([1, 1.3, 3, 3.3, 5, 5.3])
ydata = np.array([1.3, 0.7, 3.3, 2.7, 5.3, 4.7])
plt.scatter(xdata, ydata)
for (x, y, l) in zip(xdata, ydata, list("abcdef")):
    plt.annotate(l, (x+0.05, y+0.05))
# %%
X = np.vstack((xdata, ydata)).T
pca = PCA(n_components=1)
pca.fit(X)
reduced_oned = pca.fit_transform(X) # this is the "number line" view
# if you were to compress our 2D dataset down to 1D, PCA says the best way of doing that is this:
plt.scatter(reduced_oned, np.ones(6))
for (x, l) in zip(reduced_oned.ravel(), list("abcdef")):
    plt.annotate(l, (x, 1.005))
# %%
# this transform explains most of the variance while reducing the dimensionality by 1
print(f"{pca.explained_variance_ratio_ = }")
# %%
# if we were to compress along the other dimension, this wouldn't work as well
# sklearn's PCA implementation won't do this for me automatically 
# so I'm going to write a bit of it myself
pca2 = PCA(n_components=2)
pca2.fit(X)
print(f"{pca2.explained_variance_ratio_ = }")
# what do these add to? why?
# %%
# PCA gives us the principal directions of variance
plt.scatter(xdata, ydata)
# I'm putting the principal direction in green and the secondary one in red
plt.annotate("", xytext=(3.1, 3.1), xy=(3.1+pca2.components_[0,0], 3.1+pca2.components_[0,1]), arrowprops=dict(arrowstyle="->", color="green"))
plt.annotate("", xytext=(3.1, 3.1), xy=(3.1+pca2.components_[1,0], 3.1+pca2.components_[1,1]), arrowprops=dict(arrowstyle="->", color="red"))
# %%
# if we were to compress along just the green direction
# which we can do by taking a dot product between the big array X and the first component
# we'd get the same decomposition as from pca.fit_transform
my_fit_transform = np.dot(X, pca2.components_[0,:])
# the numbers are a little different because of some choices about normalization probably
# but the shape of the transformation is the same:
plt.scatter(my_fit_transform, np.ones(6))
for (x, l) in zip(my_fit_transform.ravel(), list("abcdef")):
    plt.annotate(l, (x, 1.005))
# %%
# so we now know how to compress along _any_ direction, so we could do the secondary principal one
my_secondary_fit_transform = np.dot(X, pca2.components_[1,:])
# the numbers are a little different because of some choices about normalization probably
# but the shape of the transformation is the same:
plt.scatter(my_secondary_fit_transform, np.ones(6))
for (x, l) in zip(my_secondary_fit_transform.ravel(), list("abcdef")):
    plt.annotate(l, (x, 1.005))
# is that as helpful? why/why not? 
# %%
