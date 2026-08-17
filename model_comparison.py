from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from rod_problem import RodProblem
from pinn_model import PINNModel

class ModelComparison:
  """Train, evaluate and compare multiple PINN models."""

  def __init__(
    self,
    problem: RodProblem,
    models: Sequence[PINNModel],
    number_of_test_points: int = 300,
  ) -> None:
    if not models:
      raise ValueError("At least one model is required.")

    self.problem = problem
    self.models = list(models)

    self.x_test = np.linspace(
      0,
      self.problem.length,
      number_of_test_points,
    ).reshape(-1, 1)

  def train_all(self) -> None:
    """Train every configured model."""

    for index, model in enumerate(
      self.models,
      start=1,
    ):
      print()
      print(
        f"Training model {index}/"
        f"{len(self.models)}: "
        f"{model.configuration.name}"
      )

      model.train()

  def print_summary(self) -> None:
    """Print a clean comparison table."""

    headings = (
      "Model",
      "Layers",
      "Neurons",
      "Parameters",
      "Final loss",
      "Mean error",
      "Max error",
    )

    rows = []

    for model in self.models:
      final_loss = (
        model.loss_history[-1]
        if model.loss_history
        else float("nan")
      )

      rows.append(
        (
          model.configuration.name,
          model.configuration.hidden_layers,
          model.configuration.neurons_per_layer,
          model.number_of_parameters,
          final_loss,
          model.mean_absolute_error(self.x_test),
          model.maximum_absolute_error(self.x_test),
        )
      )

    header_format = (
      "{:<18} {:>8} {:>9} {:>12} "
      "{:>14} {:>14} {:>14}"
    )

    print()
    print(header_format.format(*headings))
    print("-" * 98)

    for row in rows:
      print(
        header_format.format(
          row[0],
          row[1],
          row[2],
          row[3],
          f"{row[4]:.4e}",
          f"{row[5]:.4e}",
          f"{row[6]:.4e}",
        )
      )

  def plot_solutions(self) -> None:
    """Compare predicted and exact displacement curves."""

    exact_solution = self.problem.exact_solution(
      self.x_test
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
      self.x_test,
      exact_solution,
      linewidth=3,
      label="Exact solution",
    )

    for model in self.models:
      prediction = model.predict_physical(
        self.x_test
      )

      plt.plot(
        self.x_test,
        prediction,
        label=model.configuration.name,
      )

    plt.xlabel("x (m)")
    plt.ylabel("Displacement u(x)")
    plt.title(
      "PINN predictions compared with exact solution"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

  def plot_absolute_errors(self) -> None:
    """Plot the absolute error of every model."""

    plt.figure(figsize=(9, 6))

    for model in self.models:
      errors = model.absolute_error(
        self.x_test
      )

      plt.plot(
        self.x_test,
        errors,
        label=model.configuration.name,
      )

    plt.xlabel("x (m)")
    plt.ylabel("Absolute error")
    plt.title(
      "Absolute-error comparison"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

  def plot_loss_histories(self) -> None:
    """Compare model training-loss histories."""

    plt.figure(figsize=(9, 6))

    for model in self.models:
      plt.semilogy(
        range(
          1,
          len(model.loss_history) + 1,
        ),
        model.loss_history,
        label=model.configuration.name,
      )

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training-loss comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

  def plot_error_summary(self) -> None:
    """Display mean and maximum error by model."""

    model_names = [
      model.configuration.name
      for model in self.models
    ]

    mean_errors = [
      model.mean_absolute_error(self.x_test)
      for model in self.models
    ]

    maximum_errors = [
      model.maximum_absolute_error(self.x_test)
      for model in self.models
    ]

    positions = np.arange(
      len(self.models)
    )

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
      positions - width / 2,
      mean_errors,
      width=width,
      label="Mean absolute error",
    )

    plt.bar(
      positions + width / 2,
      maximum_errors,
      width=width,
      label="Maximum absolute error",
    )

    plt.xticks(
      positions,
      model_names,
      rotation=20,
    )

    plt.ylabel("Error")
    plt.title("Model-error summary")
    plt.grid(
      axis="y",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

  def show_all_results(self) -> None:
    """Print and visualize all comparison results."""

    self.print_summary()
    self.plot_solutions()
    self.plot_absolute_errors()
    self.plot_loss_histories()
    self.plot_error_summary()


