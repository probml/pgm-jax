
Loopy Belief Propagation for Gaussian Graphical Models in JAX.

This code is based on [this PyTorch colab](https://colab.research.google.com/drive/1-nrE95X4UC9FBLR0-cTnsIP_XhA_PZKW?usp=sharing)
by Joseph Ortiz. The translation to JAX is by Giles Harper-Donnelly.
However, it has been completely redesigned to be functionally pure, rather than object-oriented,
which makes it faster. It has also been simplified so it only works with linear Gaussian models,
and does not support iterative relinearization or robust potentials. The unit test checks that it gives the same results as Kalman smoothing on a chain-structured model.
