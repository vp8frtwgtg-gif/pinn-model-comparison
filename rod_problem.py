from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RodProblem:
  """Physical model of a rod under a linearly distributed load."""

  youngs_modulus: float = 200e9
  cross_section_area: float = 0.01
  load_coefficient: float = 1000.0
  length: float = 1.0

  @property
  def displacement_scale(self) -> float:
    return (
        self.load_coefficient
        * self.length**3
        / (
            self.cross_section_area
            * self.youngs_modulus
        )
    )

  def dimensionless_exact_solution(
      self,
      xi: np.ndarray,
  ) -> np.ndarray:
      return (
          -xi**3 + 3 * xi
      ) / 6

  def exact_solution(self, x: np.ndarray) -> np.ndarray:
    """Return the analytical displacement solution."""
    
    xi = x / self.length

    return (
        self.displacement_scale
        * self.dimensionless_exact_solution(xi)
    )
