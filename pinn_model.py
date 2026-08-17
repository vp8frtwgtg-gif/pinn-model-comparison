import numpy as np
import tensorflow as tf

from rod_problem import RodProblem
from pinn_configuration import PINNConfiguration

class PINNModel:
  """One physics-informed neural-network model."""

  def __init__(
    self,
    problem: RodProblem,
    configuration: PINNConfiguration,
    number_of_collocation_points: int = 100,
    random_seed: int | None = None,
  ) -> None:
    if number_of_collocation_points < 2:
      raise ValueError("At least two collocation points are required.")

    self.problem = problem
    self.configuration = configuration
    self.number_of_collocation_points = (number_of_collocation_points)

    if random_seed is not None:
      np.random.seed(random_seed)
      tf.random.set_seed(random_seed)

    self.x_collocation = tf.convert_to_tensor(
      np.linspace(
        0.0,
        1.0,
        self.number_of_collocation_points,
      ).reshape(-1, 1),
      dtype=tf.float32,
    )

    self.model = self._build_network()

    self.optimizer = tf.keras.optimizers.Adam(
    learning_rate=self.configuration.learning_rate
    )

    self.loss_history: list[float] = []

  def _build_network(self) -> tf.keras.Model:
    """Construct the neural-network architecture."""

    layers: list[tf.keras.layers.Layer] = [
      tf.keras.layers.Input(shape=(1,))
    ]

    for _ in range(self.configuration.hidden_layers):
      layers.append(
        tf.keras.layers.Dense(
          self.configuration.neurons_per_layer,
          activation=self.configuration.activation,
        )
      )

    layers.append(
      tf.keras.layers.Dense(
        1,
        activation=None,
      )
    )

    return tf.keras.Sequential(
      layers,
      name=self.configuration.name,
    )

  def compute_loss_components(self,) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate PDE and boundary-condition losses."""

    x = self.x_collocation

    with tf.GradientTape() as second_tape:
      second_tape.watch(x)

      with tf.GradientTape() as first_tape:
        first_tape.watch(x)
        displacement = self.model(
          x,
          training=True,
        )

      first_derivative = first_tape.gradient(
        displacement,
        x,
      )

      if first_derivative is None:
        raise RuntimeError(
            "TensorFlow could not calculate du/dx."
        )

    second_derivative = second_tape.gradient(
      first_derivative,
      x,
    )

    if second_derivative is None:
      raise RuntimeError("TensorFlow could not calculate d²u/dx².")

    pde_residual = second_derivative + x

    pde_loss = tf.reduce_mean(
      tf.square(pde_residual)
    )

    x_zero = tf.constant(
      [[0.0]],
      dtype=tf.float32,
    )

    displacement_at_zero = self.model(
      x_zero,
      training=True,
    )

    displacement_boundary_loss = tf.reduce_mean(
      tf.square(displacement_at_zero)
    )

    x_length = tf.constant(
      [[1.0]],
      dtype=tf.float32,
    )

    with tf.GradientTape() as boundary_tape:
      boundary_tape.watch(x_length)

      displacement_at_length = self.model(
        x_length,
        training=True,
      )

    derivative_at_length = boundary_tape.gradient(
      displacement_at_length,
      x_length,
    )

    if derivative_at_length is None:
      raise RuntimeError("TensorFlow could not calculate du/dx at x=L.")

    derivative_boundary_loss = tf.reduce_mean(
      tf.square(derivative_at_length)
    )

    return (
      pde_loss,
      displacement_boundary_loss,
      derivative_boundary_loss,
    )
  
  def compute_total_loss(self) -> tf.Tensor:
    """Return the complete PINN loss."""

    pde_loss, boundary_loss_1, boundary_loss_2 = (
      self.compute_loss_components()
    )

    return (
      pde_loss
      + boundary_loss_1
      + boundary_loss_2
    )

  @tf.function
  def _train_step(self) -> tf.Tensor:
    """Perform one optimizer step."""

    with tf.GradientTape() as tape:
      loss = self.compute_total_loss()

    gradients = tape.gradient(
      loss,
      self.model.trainable_variables,
    )

    gradient_variable_pairs = [
      (gradient, variable)
      for gradient, variable in zip(
        gradients,
        self.model.trainable_variables,
      )
      if gradient is not None
    ]

    self.optimizer.apply_gradients(
      gradient_variable_pairs
    )

    return loss

  def train(
    self,
    print_interval: int = 500,
  ) -> None:
    """Train the model and save its loss history."""

    self.loss_history.clear()

    for epoch in range(
      1,
      self.configuration.epochs + 1,
    ):
      loss = self._train_step()
      loss_value = float(loss.numpy())

      self.loss_history.append(loss_value)

      if (
        print_interval > 0
        and (
          epoch == 1
          or epoch % print_interval == 0
          or epoch == self.configuration.epochs
        )
      ):
        print(
          f"{self.configuration.name}: "
          f"epoch {epoch}/"
          f"{self.configuration.epochs}, "
          f"loss = {loss_value:.6e}"
        )

  def predict(
    self,
    xi: np.ndarray,
  ) -> np.ndarray:
    """Predict dimensionless displacement values."""

    xi_tensor = tf.convert_to_tensor(
      xi,
      dtype=tf.float32,
    )

    predictions = self.model(
      xi_tensor,
      training=False,
    )

    return predictions.numpy()

  def predict_physical(
      self,
      x: np.ndarray,
  ) -> np.ndarray:

    """Predict physical displacement in metres."""
    
    xi = x / self.problem.length

    dimensionless_prediction = self.predict(xi)

    return (
        self.problem.displacement_scale
        * dimensionless_prediction
    )

  def absolute_error(
    self,
    x: np.ndarray,
  ) -> np.ndarray:
    """Return absolute error against the exact solution."""

    prediction = self.predict_physical(x)
    exact = self.problem.exact_solution(x)

    return np.abs(exact - prediction)

  def mean_absolute_error(
    self,
    x: np.ndarray,
  ) -> float:
    """Return mean absolute error."""

    return float(
      np.mean(self.absolute_error(x))
    )

  def maximum_absolute_error(
    self,
    x: np.ndarray,
  ) -> float:
    """Return maximum absolute error."""

    return float(
      np.max(self.absolute_error(x))
    )

  @property
  def number_of_parameters(self) -> int:
    """Return the number of trainable model parameters."""

    return int(
      self.model.count_params()
    )
