from abc import ABC, abstractmethod
from typing import Dict

from torch import nn


# TODO: name it layer?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    def __init__(self, use_em: bool = True) -> None:
        """Init class.

        Args:
            use_em (bool, optional): Whether to use EM. Defaults to True.
        """
        super().__init__()  # TODO: do we need multi-inherit init?
        self._use_em = use_em
        self.prob = None  # TODO: type?

        # TODO: type and init for the following?
        self._online_em_counter = 0
        self.online_em_frequency = 0
        self.online_em_stepsize = 0

    @abstractmethod
    def default_initializer(self):  # TODO: return type?
        """Produce suitable initial parameters for the layer.

        :return: initial parameters
        """

    @abstractmethod
    def initialize(self, initializer=None) -> None:  # TODO: type?
        """Initialize the layer, e.g. with return value from default_initializer(self).

        :param initializer: 'default', or custom (typically a Tensor)
                            'default' means that the layer simply calls its own \
                                default_initializer(self), in stores
                            the parameters internally.
                            custom (typically a Tensor) means that you pass your own initializer.
        :return: None
        """

    @abstractmethod
    def forward(self, x=None):  # TODO: type?
        """Compute the layer. The result is always a tensor of \
            log-densities of shape (batch_size, num_dist, num_nodes), \
        where num_dist is the vector length (K in the paper) and num_nodes is \
            the number of PC nodes in the layer.

        :param x: input data (Tensor).
                  If self.num_dims == 1, this can be either of shape \
                    (batch_size, self.num_var, 1) or
                  (batch_size, self.num_var).
                  If self.num_dims > 1, this must be of shape (batch_size, \
                    self.num_var, self.num_dims).
                  Not all layers use this argument.
        :return: log-density tensor of shape (batch_size, num_dist, num_nodes), \
            where num_dist is the vector length
                 (K in the paper) and num_nodes is the number of PC nodes in the layer.
        """

    @abstractmethod
    def backtrack(self, *args, **kwargs):  # TODO: type?
        """Define routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """

    def em_set_hyperparams(
        self, online_em_frequency, online_em_stepsize, purge: bool = True
    ) -> None:  # TODO: type?
        """Set new setting for online EM.

        :param online_em_frequency: How often, i.e. after how many calls to \
            em_process_batch(self), shall em_update(self) be called?
        :param online_em_stepsize: step size of online em.
        :param purge: discard current learn statistics?
        """
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self.online_em_frequency = online_em_frequency
        self.online_em_stepsize = online_em_stepsize

    # TODO: this is not used? and em_set_params undefined
    # def em_set_batch(self):
    #     """Set batch mode EM."""
    #     self.em_set_params(None, None)

    @abstractmethod
    def em_purge(self) -> None:
        """Discard accumulated EM statistics."""
        raise NotImplementedError

    @abstractmethod
    def em_process_batch(self):  # TODO: type?
        """Process the current batch. This should be called after backwards() on the whole model."""

    @abstractmethod
    def em_update(self):  # TODO: type?
        """Perform an EM update step."""

    @abstractmethod
    def project_params(self, params):  # TODO: type?
        """Project paramters onto feasible set."""

    @abstractmethod
    def reparam_function(self):  # TODO: type?
        """Return a function which transforms a tensor of unconstrained values \
            into feasible parameters."""

    @abstractmethod
    def get_shape_dict(self) -> Dict[str, int]:
        """Return param shape of the layer with description."""
