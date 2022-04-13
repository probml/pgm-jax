import jax.numpy as jnp
from gaussian import Gaussian

class VariableNode:
    def __init__(self, id: int, dofs: int, properties: dict = {}) -> None:
        self.variableID = id
        self.properties = properties
        self.dofs = dofs
        self.adj_factors = []
        self.belief = Gaussian(dofs)
        self.prior = Gaussian(dofs)  # prior factor, implemented as part of variable node

    def update_belief(self) -> None:
        """ Update local belief estimate by taking product of all incoming messages along all edges. """
        self.belief.eta = self.prior.eta.clone()  # message from prior factor
        self.belief.lam = self.prior.lam.clone()
        for factor in self.adj_factors:  # messages from other adjacent variables
            message_ix = factor.adj_vIDs.index(self.variableID)
            self.belief.eta += factor.messages[message_ix].eta
            self.belief.lam += factor.messages[message_ix].lam

    def get_prior_energy(self) -> float:
        energy = 0.
        if self.prior.lam[0, 0] != 0.:
            residual = self.belief.mean() - self.prior.mean()
            energy += 0.5 * residual @ self.prior.lam @ residual
        return energy

