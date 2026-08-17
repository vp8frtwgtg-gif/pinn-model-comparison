from mmodel_comparison import ModelComparison
from pinn_model import PINNModel
from rod_problem import RodProblem
from user_input import read_model_configurations

def main() -> None:
  """Create, train and compare user-configured PINNs."""

  problem = RodProblem()
  configurations = read_model_configurations()

  models = [
    PINNModel(
      problem=problem,
      configuration=configuration,
      number_of_collocation_points=100,
      random_seed=42 + index,
    )
    for index, configuration in enumerate(
      configurations
    )
  ]

  comparison = ModelComparison(
    problem=problem,
    models=models,
  )

  comparison.train_all()
  comparison.show_all_results()


if __name__ == "__main__":
  main()
