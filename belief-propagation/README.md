
Loopy Belief Propagation for Gaussian Graphical Models in JAX.

The code is based on [this PyTorch colab](https://colab.research.google.com/drive/1-nrE95X4UC9FBLR0-cTnsIP_XhA_PZKW?usp=sharing)
by Joseph Ortiz, which is explained at https://gaussianbp.github.io/.
This allows for nonlinear factors by iteratively linearizing the factors.
Thus the MAP estimate corresponds to solving a nonlinear least squares problem.
The translation to JAX is by moloydas@, murphyk@.

