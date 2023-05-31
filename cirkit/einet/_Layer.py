from typing import Dict

import torch


class Layer(torch.nn.Module):
    """
    Abstract layer class. Specifies functionality every layer in an EiNet should implement.
    """

    def __init__(self, use_em=True):
        super(Layer, self).__init__()
        self._use_em = use_em
        self.prob = None

    def default_initializer(self):
        """
        Produce suitable initial parameters for the layer.
        :return: initial parameters
        """
        raise NotImplementedError

    def initialize(self, initializer=None):
        """
        Initialize the layer, e.g. with return value from default_initializer(self).
        :param initializer: 'default', or custom (typically a Tensor)
                            'default' means that the layer simply calls its own default_initializer(self), in stores
                            the parameters internally.
                            custom (typically a Tensor) means that you pass your own initializer.
        :return: None
        """
        raise NotImplementedError

    def forward(self, x=None):
        """
        Compute the layer. The result is always a tensor of log-densities of shape (batch_size, num_dist, num_nodes),
        where num_dist is the vector length (K in the paper) and num_nodes is the number of PC nodes in the layer.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, self.num_var, self.num_dims).
                  Not all layers use this argument.
        :return: log-density tensor of shape (batch_size, num_dist, num_nodes), where num_dist is the vector length
                 (K in the paper) and num_nodes is the number of PC nodes in the layer.
        """
        raise NotImplementedError

    def backtrack(self, *args, **kwargs):
        """
        Defines routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def em_set_hyperparams(self, online_em_frequency, online_em_stepsize, purge=True):
        """
        Set new setting for online EM.

        :param online_em_frequency: How often, i.e. after how many calls to em_process_batch(self), shall
                                    em_update(self) be called?
        :param online_em_stepsize: step size of online em.
        :param purge: discard current learn statistics?
        :return: None
        """
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self.online_em_frequency = online_em_frequency
        self.online_em_stepsize = online_em_stepsize

    def em_set_batch(self):
        """Set batch mode EM."""
        self.em_set_params(None, None)

    def em_purge(self):
        """Discard accumulated EM statistics """
        raise NotImplementedError

    def em_process_batch(self):
        """Process the current batch. This should be called after backwards() on the whole model."""
        raise NotImplementedError

    def em_update(self):
        """Perform an EM update step."""
        raise NotImplementedError

    def project_params(self, params):
        """Project paramters onto feasible set."""
        raise NotImplementedError

    def reparam_function(self):
        """Return a function which transforms a tensor of unconstrained values into feasible parameters."""
        raise NotImplementedError

    def get_shape_dict(self) -> Dict[str, int]:
        """Return param shape of the layer with description"""
        raise NotImplementedError

