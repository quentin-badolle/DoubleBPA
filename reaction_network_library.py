"""
A collection of Stochastic Chemical Reaction Networks
"""

import os
import numpy as np
import itertools

try:
    import jax.numpy as jnp
except ImportError:
    print("jax operations won't be supported")


import reaction_network_definition as rxn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class ConstitutiveGeneExpression(rxn.ReactionNetwork):
    """Defines the gene expression network.

    Parameters from "Unbiased Estimation of Parameter Sensitivities for Stochastic Chemical Reaction Networks" by Gupta et al.
    citing "Intrinsic noise in gene regulatory networks" by Thattai and A. van Oudenaarden.
    """

    def __init__(self):
        crn_name = "constitutive_gene_expression"
        num_species = 2
        initial_state = np.array([0.0, 0.0])
        num_reactions = 4
        reactant_matrix = np.zeros([num_species, num_reactions], dtype=int)
        product_matrix = np.zeros([num_species, num_reactions], dtype=int)
        # 1. 0 --> M
        product_matrix[0, 0] = 1
        # 2. M --> M + P
        reactant_matrix[0, 1] = 1
        product_matrix[0, 1] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[0, 2] = 1
        # 4. P -->0
        reactant_matrix[1, 3] = 1
        parameter_dict = {
            "transcription rate": 0.6,
            "translation rate": 1.7329,
            "mRNA degradation rate": 0.3466,
            "protein degradation rate": 0.0023,
        }
        reaction_dict = {
            0: ["mass action", "transcription rate"],
            1: ["mass action", "translation rate"],
            2: ["mass action", "mRNA degradation rate"],
            3: ["mass action", "protein degradation rate"],
        }
        species_labels = ["mRNA", "protein"]
        output_species_labels = ["protein"]
        output_labels = ["protein"]
        time_unit = "min."

        super(ConstitutiveGeneExpression, self).__init__(
            crn_name,
            num_species,
            initial_state,
            num_reactions,
            reactant_matrix,
            product_matrix,
            parameter_dict,
            reaction_dict,
            species_labels,
            output_species_labels,
            output_labels,
            time_unit,
        )

        self.output_function_size = 1

    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]

        return np.stack(output_list, axis=1).flatten()

    def exact_first_moments_with_explicit_expression(
        self, param_values, initial_state, final_time
    ):
        first_moments = jnp.array(
            [
                initial_state[0] * jnp.exp(-final_time * param_values[2])
                + param_values[0]
                / param_values[2]
                * (1 - jnp.exp(-final_time * param_values[2])),
                initial_state[1] ** 2 * jnp.exp(-final_time * param_values[3])
                + param_values[1]
                * jnp.exp(-final_time * param_values[3])
                * (
                    initial_state[0]
                    * (jnp.exp(final_time * (param_values[3] - param_values[2])) - 1)
                    / (param_values[3] - param_values[2])
                    + param_values[0]
                    / param_values[2]
                    * (
                        1.0
                        / param_values[3]
                        * (jnp.exp(final_time * param_values[3]) - 1)
                        - (
                            jnp.exp(final_time * (param_values[3] - param_values[2]))
                            - 1.0
                        )
                        / (param_values[3] - param_values[2])
                    )
                ),
                initial_state[0] ** 2 * jnp.exp(-2 * final_time * param_values[2])
                + (2 * param_values[0] + param_values[2])
                / param_values[2]
                * initial_state[0]
                * (
                    jnp.exp(-final_time * param_values[2])
                    - jnp.exp(-2 * final_time * param_values[2])
                )
                + (2 * param_values[0] + param_values[2])
                * param_values[0]
                / param_values[2] ** 2
                * (
                    1 / 2
                    - jnp.exp(-final_time * param_values[2])
                    + jnp.exp(-2 * final_time * param_values[2]) / 2
                )
                + param_values[0]
                / (2 * param_values[2])
                * (1 - jnp.exp(-2 * final_time * param_values[2])),
            ]
        )
        return first_moments


class ToggleSwitch(rxn.ReactionNetwork):
    """Defines the genetic toggle switch."""

    def __init__(self):
        crn_name = "toggle_switch"
        num_species = 2
        initial_state = np.array([0.0, 0.0])
        num_reactions = 4
        reactant_matrix = np.zeros([num_species, num_reactions], dtype=int)
        product_matrix = np.zeros([num_species, num_reactions], dtype=int)
        # 1. 0 --> S_1
        product_matrix[0, 0] = 1
        # 2. S_1 --> 0
        reactant_matrix[0, 1] = 1
        # 3. 0 --> S_2
        product_matrix[1, 2] = 1
        # 4. S_2 --> 0
        reactant_matrix[1, 3] = 1
        parameter_dict = {
            "alpha_1": 1.0,
            "k_1": 1.0,
            "beta": 2.5,
            "b_1": 0.5,
            "mak_1": 0.0023,
            "alpha_2": 1.0,
            "k_2": 1.0,
            "gamma": 1.0,
            "b_2": 0.5,
            "mak_2": 0.0023,
        }
        reaction_dict = {
            0: ["hill repression", 1, "alpha_1", "k_1", "beta", "b_1"],
            1: ["mass action", "mak_1"],
            2: ["hill repression", 0, "alpha_2", "k_2", "gamma", "b_2"],
            3: ["mass action", "mak_2"],
        }
        species_labels = ["species_1", "species_2"]
        output_species_labels = ["species_1"]
        output_labels = ["species_1"]
        time_unit = "min."

        super(ToggleSwitch, self).__init__(
            crn_name,
            num_species,
            initial_state,
            num_reactions,
            reactant_matrix,
            product_matrix,
            parameter_dict,
            reaction_dict,
            species_labels,
            output_species_labels,
            output_labels,
            time_unit,
        )

        self.output_function_size = 1

    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]

        return np.stack(output_list, axis=1).flatten()


class AntitheticGeneExpression(rxn.ReactionNetwork):
    """Define the antithetic integral controller."""

    def __init__(self):
        crn_name = "antithetic_gene_expression"
        num_species = 4
        initial_state = np.array([0.0, 0.0, 0.0, 0.0])
        num_reactions = 7
        species_labels = ["mRNA", "protein", "Z1", "Z2"]
        output_species_labels = ["protein"]
        reactant_matrix = np.zeros([num_reactions, num_species], dtype=int)
        product_matrix = np.zeros([num_reactions, num_species], dtype=int)
        # 1. Z_1 --> Z_1 + M
        product_matrix[0, 2] = 1
        product_matrix[0, 0] = 1
        reactant_matrix[0, 2] = 1
        # 2. M --> M + P
        reactant_matrix[1, 0] = 1
        product_matrix[1, 0] = 1
        product_matrix[1, 1] = 1
        # 3. M --> 0
        reactant_matrix[2, 0] = 1
        # 4. P -->0
        reactant_matrix[3, 1] = 1
        # 5. P -->P + Z_2
        reactant_matrix[4, 1] = 1
        product_matrix[4, 1] = 1
        product_matrix[4, 3] = 1
        # 6. Z_1 + Z_2 -->0
        reactant_matrix[5, 2] = 1
        reactant_matrix[5, 3] = 1
        # 7. 0 --> Z_1
        product_matrix[6, 2] = 1
        reactant_matrix = reactant_matrix.T
        product_matrix = product_matrix.T
        parameter_dict = {
            "activation rate": 0.6,
            "translation rate": 1.7329,
            "mRNA degradation rate": 0.3466,
            "protein degradation rate": 0.0023,
            "theta": 0.1,
            "eta": 10.0,
            "mu": 0.5,
        }
        reaction_dict = {
            0: ["mass action", "activation rate"],
            1: ["mass action", "translation rate"],
            2: ["mass action", "mRNA degradation rate"],
            3: ["mass action", "protein degradation rate"],
            4: ["mass action", "theta"],
            5: ["mass action", "eta"],
            6: ["mass action", "mu"],
        }
        self.species_labels = ["mRNA", "protein", "Z1", "Z2"]
        output_species_labels = ["protein"]
        output_labels = ["protein", "protein^2"]
        time_unit = "min."

        super(AntitheticGeneExpression, self).__init__(
            crn_name,
            num_species,
            initial_state,
            num_reactions,
            reactant_matrix,
            product_matrix,
            parameter_dict,
            reaction_dict,
            species_labels,
            output_species_labels,
            output_labels,
            time_unit,
        )

        self.output_function_size = 2

    def output_function(self, state):
        output_list = [state[:, i] for i in self.output_species_indices]
        output_list_second_moment = [
            state[:, i] ** 2 for i in self.output_species_indices
        ]
        for elem in output_list_second_moment:
            output_list.append(elem)

        return np.stack(output_list, axis=1).flatten()
