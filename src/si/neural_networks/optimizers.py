from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):
    """
    Adam optimizer.

    Adam (Adaptive Moment Estimation) combines momentum and adaptive
    learning rates to achieve faster and more stable convergence.
    It keeps track of exponentially decaying averages of past gradients
    (first moment) and squared gradients (second moment).
    """

    def __init__(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate : float
            Step size used to update the parameters.
        beta_1 : float, optional
            Exponential decay rate for the first moment estimates (default=0.9).
        beta_2 : float, optional
            Exponential decay rate for the second moment estimates (default=0.999).
        epsilon : float, optional
            Small constant added for numerical stability (default=1e-8).
        """
        super().__init__(learning_rate)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # First and second moment vectors
        self.m = None
        self.v = None

        # Time step
        self.t = 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update weights using the Adam optimization algorithm.

        Parameters
        ----------
        w : np.ndarray
            Current weights of the model.
        grad_loss_w : np.ndarray
            Gradient of the loss function with respect to the weights.

        Returns
        -------
        np.ndarray
            Updated weights after applying the Adam update rule.
        """
        # Initialize moment vectors if this is the first update
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        # Increment time step
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Update biased second moment estimate
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Update parameters
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)