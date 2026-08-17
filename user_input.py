from pinn_configuration import PINNConfiguration

def read_positive_integer(
  prompt: str,
) -> int:
  """Read a positive integer from the user."""

  while True:
    try:
      value = int(input(prompt))

      if value < 1:
        raise ValueError

      return value

    except ValueError:
      print(
        "Please enter a positive whole number."
      )


def read_positive_float(
  prompt: str,
) -> float:
  """Read a positive floating-point number."""

  while True:
    try:
      value = float(input(prompt))

      if value <= 0:
        raise ValueError

      return value

    except ValueError:
      print(
        "Please enter a positive number."
      )

def read_activation_function() -> str:
  """Ask for an activation suitable for a second-order PINN."""

  supported_activations = {
    "tanh",
    "sigmoid",
    "swish",
    "softplus",
  }

  while True:
    activation = input(
      "Activation function "
      "(tanh, sigmoid, swish, softplus): "
    ).strip().lower()

    if not activation:
      return "tanh"

    if activation in supported_activations:
      return activation

    print(
      "Please choose tanh, sigmoid, swish, or softplus."
    )

def read_model_configurations(
) -> list[PINNConfiguration]:
  """Ask the user which network architectures to compare."""

  number_of_models = read_positive_integer(
    "How many models do you want to compare? "
  )

  configurations = []

  for index in range(
    1,
    number_of_models + 1,
  ):
    print()
    print(f"Configuration for model {index}")

    default_name = f"Model {index}"

    name = input(
      f"Model name [{default_name}]: "
    ).strip()

    if not name:
      name = default_name

    #Make name TensorFlow safe
    name = name.replace(" ", "_")

    if not name[0].isalpha():
      name = f"model_{name}"

    hidden_layers = read_positive_integer(
      "Number of hidden layers: "
    )

    neurons_per_layer = read_positive_integer(
      "Neurons per hidden layer: "
    )

    activation = read_activation_function()

    if not activation:
      activation = "tanh"

    learning_rate = read_positive_float(
      "Learning rate, for example 0.001: "
    )

    epochs = read_positive_integer(
      "Number of epochs: "
    )

    configurations.append(
      PINNConfiguration(
        name=name,
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons_per_layer,
        activation=activation,
        learning_rate=learning_rate,
        epochs=epochs,
      )
    )

  return configurations
