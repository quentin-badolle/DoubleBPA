"""
A collection of propensities and their derivatives.
"""

import math
import numpy as np


class Propensity(object):
    """
    A collection of propensities.
    """

    def __init__(
        self, num_species: int, reactant_matrix: np.array, parameter_dict: dict
    ):
        self.num_species = num_species
        self.reactant_matrix = reactant_matrix
        self.parameter_dict = parameter_dict

    def mass_action(
        self, state: np.array, reaction_no: int, rate_constant_key: str
    ) -> float:
        """Computes a mass action propensity."""
        prop_elem = self.parameter_dict[rate_constant_key]
        for j in range(self.num_species):
            for k in range(self.reactant_matrix[j][reaction_no]):
                prop_elem *= float(state[j] - k)
            prop_elem = prop_elem / math.factorial(self.reactant_matrix[j][reaction_no])
        return prop_elem

    def hill_activation(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
    ) -> float:
        """Computes the Hill activation propensity b + a * x^h / (k + x^h) with x=X[species_no]."""
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        prop_elem = b + a * (xp**h) / (k + (xp**h))
        return prop_elem

    def hill_repression(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
    ) -> float:
        """Computes the Hill repression propensity b + a / (k + x^h) with x=X[species_no]."""
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        xp = float(state[species_no])
        prop_elem = b + a / (k + (xp**h))
        return prop_elem

    def linear(
        self, state: np.array, species_no: int, parameter_key1: str, parameter_key2: str
    ) -> float:
        """Computes the linear propensity max(a + b*x, 0) with x=X[species_no]."""
        a = self.parameter_dict[parameter_key1]
        b = self.parameter_dict[parameter_key2]
        xp = float(state[species_no])
        prop_elem = max(a + b * xp, 0)
        return prop_elem


class PropensityFirstOrderDerivative(object):
    """
    A collection of first order derivatives of propensities.
    """

    def __init__(
        self, num_species: int, reactant_matrix: np.array, parameter_dict: dict
    ):
        self.num_species = num_species
        self.reactant_matrix = reactant_matrix
        self.parameter_dict = parameter_dict

    def mass_action(
        self,
        state: np.array,
        reaction_no: int,
        rate_constant_key: str,
        param_names: list,
    ) -> list[float]:
        """Computes the first order derivative of propensities w.r.t. the parameter of one mass action propensity."""
        propensity_derivatives = np.zeros([len(param_names)])
        der_elem = 1
        for j in range(self.num_species):
            for k in range(self.reactant_matrix[j][reaction_no]):
                der_elem *= float(state[j] - k)
            der_elem = der_elem / math.factorial(self.reactant_matrix[j][reaction_no])
        propensity_derivatives[param_names.index(rate_constant_key)] = der_elem
        return propensity_derivatives

    def hill_activation(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ) -> list[float]:
        """Computes the first order derivative of propensities w.r.t. the parameters of one Hill activation propensity
        b + a * x^h / (k + x^h) with x=X[species_no].
        """
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        den = k + (xp**h)
        propensity_derivatives = np.zeros([len(param_names)])
        propensity_derivatives[param_names.index(parameter_key1)] = (xp**h) / den
        propensity_derivatives[param_names.index(parameter_key2)] = (
            -a * (xp**h) / (den**2)
        )
        if xp > 0:
            propensity_derivatives[param_names.index(parameter_key3)] = (
                a * k * (xp**h) * math.log(xp)
            ) / (
                den**2
            )  # is there a - really? was removed
        if parameter_key4 is not None:
            propensity_derivatives[param_names.index(parameter_key4)] = 1.0
        return propensity_derivatives

    def hill_repression(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ) -> list[float]:
        """Computes the first order derivative of propensities w.r.t. the parameters of one Hill repression propensity
        b + a / (k + x^h) with x=X[species_no].
        """
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        den = k + (xp**h)
        propensity_derivatives = np.zeros([len(param_names)])
        propensity_derivatives[param_names.index(parameter_key1)] = 1 / den
        propensity_derivatives[param_names.index(parameter_key2)] = -a / (den**2)
        if xp > 0:
            propensity_derivatives[param_names.index(parameter_key3)] = -(
                a * (xp**h) * math.log(xp)
            ) / (den**2)
        if parameter_key4 is not None:
            propensity_derivatives[param_names.index(parameter_key4)] = 1.0
        return propensity_derivatives


class LogPropensityFirstOrderDerivative(object):
    """
    A collection of first order derivatives of log propensities.
    """

    def __init__(
        self, num_species: int, reactant_matrix: np.array, parameter_dict: dict
    ):
        self.num_species = num_species
        self.reactant_matrix = reactant_matrix
        self.parameter_dict = parameter_dict
        self.prop_deriv = PropensityFirstOrderDerivative(
            num_species, reactant_matrix, parameter_dict
        )

    def mass_action(
        self,
        rate_constant_key: str,
        param_names: list,
    ):
        propensity_derivatives = np.zeros([len(param_names)])
        propensity_derivatives[param_names.index(rate_constant_key)] = (
            1 / self.parameter_dict[rate_constant_key]
        )
        return propensity_derivatives

    def hill_activation(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ) -> list[float]:
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        func = b + a * (xp**h) / (k + (xp**h))
        propensity_derivatives = self.prop_deriv.hill_activation(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_derivatives = propensity_derivatives / func
        return propensity_derivatives

    def hill_repression(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ) -> list[float]:
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        func = b + a / (k + (xp**h))
        propensity_derivatives = self.prop_deriv.hill_repression(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_derivatives = propensity_derivatives / func
        return propensity_derivatives


class PropensitySecondOrderDerivative(object):
    """
    A collection of second order derivatives of propensities.
    """

    def __init__(
        self, num_species: int, reactant_matrix: np.array, parameter_dict: dict
    ):
        self.num_species = num_species
        self.reactant_matrix = reactant_matrix
        self.parameter_dict = parameter_dict

    def mass_action(
        self,
        param_names: list,
    ):
        return np.zeros([len(param_names), len(param_names)])

    def hill_activation(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ):
        a = self.parameter_dict[parameter_key1]
        idx_a = param_names.index(parameter_key1)
        k = self.parameter_dict[parameter_key2]
        idx_k = param_names.index(parameter_key2)
        h = self.parameter_dict[parameter_key3]
        idx_h = param_names.index(parameter_key3)
        xp = float(state[species_no])
        den = k + (xp**h)
        propensity_second_derivatives = np.zeros([len(param_names), len(param_names)])
        propensity_second_derivatives[idx_a, idx_k] = -(xp**h) / den**2
        propensity_second_derivatives[idx_k, idx_a] = -(xp**h) / den**2
        propensity_second_derivatives[idx_k, idx_k] = 2 * a * (xp**h) / den**3
        if xp > 0:
            propensity_second_derivatives[idx_a, idx_h] = (
                k * xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_h, idx_a] = (
                k * xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_k, idx_h] = (
                a * xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_h, idx_k] = (
                a * xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_h, idx_h] = (
                a * k * xp**h * (math.log(xp) ** 2) * (k - xp**h) / den**3
            )
        return propensity_second_derivatives

    def hill_repression(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ):
        a = self.parameter_dict[parameter_key1]
        idx_a = param_names.index(parameter_key1)
        k = self.parameter_dict[parameter_key2]
        idx_k = param_names.index(parameter_key2)
        h = self.parameter_dict[parameter_key3]
        idx_h = param_names.index(parameter_key3)
        xp = float(state[species_no])
        den = k + (xp**h)
        propensity_second_derivatives = np.zeros([len(param_names), len(param_names)])
        propensity_second_derivatives[idx_a, idx_k] = -1 / den**2
        propensity_second_derivatives[idx_k, idx_a] = -1 / den**2
        propensity_second_derivatives[idx_k, idx_k] = 2 * a / den**3
        if xp > 0:
            propensity_second_derivatives[idx_a, idx_h] = -(
                xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_h, idx_a] = -(
                xp**h * math.log(xp) / den**2
            )
            propensity_second_derivatives[idx_k, idx_h] = (
                2 * a * xp**h * math.log(xp) / den**3
            )
            propensity_second_derivatives[idx_h, idx_k] = (
                2 * a * xp**h * math.log(xp) / den**3
            )
            propensity_second_derivatives[idx_h, idx_h] = (
                a * xp**h * (math.log(xp) ** 2) * (xp**h - k) / den**3
            )
        return propensity_second_derivatives


class LogPropensitySecondOrderDerivative(object):
    """
    A collection of second order derivatives of log propensities.
    """

    def __init__(
        self, num_species: int, reactant_matrix: np.array, parameter_dict: dict
    ):
        self.num_species = num_species
        self.reactant_matrix = reactant_matrix
        self.parameter_dict = parameter_dict
        self.prop_first_deriv = PropensityFirstOrderDerivative(
            num_species, reactant_matrix, parameter_dict
        )
        self.prop_second_deriv = PropensitySecondOrderDerivative(
            num_species, reactant_matrix, parameter_dict
        )

    def mass_action(
        self,
        rate_constant_key: str,
        param_names: list,
    ):
        propensity_derivatives = np.zeros([len(param_names), len(param_names)])
        propensity_derivatives[
            param_names.index(rate_constant_key), param_names.index(rate_constant_key)
        ] = (-1 / self.parameter_dict[rate_constant_key] ** 2)
        return propensity_derivatives

    def hill_activation(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ):
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        func = b + a * (xp**h) / (k + (xp**h))
        propensity_second_derivatives = self.prop_second_deriv.hill_activation(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_second_derivatives = propensity_second_derivatives / func
        proprensity_first_derivatives = self.prop_first_deriv.hill_activation(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_second_derivatives = (
            propensity_second_derivatives
            - np.outer(proprensity_first_derivatives, proprensity_first_derivatives)
            / func**2
        )
        return propensity_second_derivatives

    def hill_repression(
        self,
        state: np.array,
        species_no: int,
        parameter_key1: str,
        parameter_key2: str,
        parameter_key3: str,
        parameter_key4: str,
        param_names: list,
    ):
        a = self.parameter_dict[parameter_key1]
        k = self.parameter_dict[parameter_key2]
        h = self.parameter_dict[parameter_key3]
        xp = float(state[species_no])
        if parameter_key4 is not None:
            b = self.parameter_dict[parameter_key4]
        else:
            b = 0.0
        func = b + a / (k + (xp**h))
        propensity_second_derivatives = self.prop_second_deriv.hill_repression(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_second_derivatives = propensity_second_derivatives / func
        proprensity_first_derivatives = self.prop_first_deriv.hill_repression(
            state,
            species_no,
            parameter_key1,
            parameter_key2,
            parameter_key3,
            parameter_key4,
            param_names,
        )
        propensity_second_derivatives = (
            propensity_second_derivatives
            - np.outer(proprensity_first_derivatives, proprensity_first_derivatives)
            / func**2
        )
        return propensity_second_derivatives
