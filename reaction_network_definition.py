"""
Stochastic Chemical Reaction Network simulation environment.
"""

import numpy as np
import math
from tqdm import tqdm
import time

import propensity_library as prop_lib


class ReactionNetwork(object):
    """Reference class to perform the BPA and Girsanov transform method for second-order sensitivities."""

    def __init__(
        self,
        crn_name,
        num_species: int,
        initial_state: np.array,
        num_reactions: int,
        reactant_matrix: np.array,
        product_matrix: np.array,
        parameter_dict: dict,
        reaction_dict: dict,
        species_labels: list,
        output_species_labels: list,
        output_labels: str,
        time_unit: str,
    ):

        self.crn_name = crn_name
        self.num_species = num_species
        self.initial_state = initial_state
        self.num_reactions = num_reactions
        # reactant_matrix rows represent nb of molecules of each species consumed in that reaction
        self.reactant_matrix = reactant_matrix
        # product_matrix rows represent nb of molecules of each species produced in that reaction
        self.product_matrix = product_matrix
        self.stoichiometry_matrix = product_matrix - reactant_matrix
        # parameter_dict contains information about the parameters
        self.parameter_dict = parameter_dict
        self.reaction_dict = reaction_dict
        self.species_labels = species_labels
        self.output_species_labels = output_species_labels
        self.output_species_indices = [
            self.species_labels.index(i) for i in self.output_species_labels
        ]
        self.output_labels = output_labels
        self.output_function_size = None
        self.time_unit = time_unit

    ### Shared utilities between the BPA and the Girsanov transform method for second-order sensitivities

    def propensity_vector(self, state: np.array) -> np.array:
        """
        Computes the whole propensity vector.

        Dimension of the output: (nb of reactions,).
        """
        propensity = prop_lib.Propensity(
            self.num_species, self.reactant_matrix, self.parameter_dict
        )
        prop = np.zeros(self.num_reactions)
        # Fill the propensity vector one element (i.e. one reaction) at a time
        for reaction_no in range(self.num_reactions):
            reaction_type = self.reaction_dict[reaction_no][0]
            if reaction_type == "mass action":
                prop[reaction_no] = propensity.mass_action(
                    state, reaction_no, self.reaction_dict[reaction_no][1]
                )
            elif reaction_type == "hill activation":
                prop[reaction_no] = propensity.hill_activation(
                    state,
                    self.reaction_dict[reaction_no][1],
                    self.reaction_dict[reaction_no][2],
                    self.reaction_dict[reaction_no][3],
                    self.reaction_dict[reaction_no][4],
                    self.reaction_dict[reaction_no][5],
                )
            elif reaction_type == "hill repression":
                prop[reaction_no] = propensity.hill_repression(
                    state,
                    self.reaction_dict[reaction_no][1],
                    self.reaction_dict[reaction_no][2],
                    self.reaction_dict[reaction_no][3],
                    self.reaction_dict[reaction_no][4],
                    self.reaction_dict[reaction_no][5],
                )
            elif reaction_type == "linear":
                prop[reaction_no] = propensity.linear(
                    state,
                    self.reaction_dict[reaction_no][1],
                    self.reaction_dict[reaction_no][2],
                    self.reaction_dict[reaction_no][3],
                )
            else:
                raise NotImplementedError
        return prop

    def propensity_first_order_derivative(self, state: np.array) -> np.array:
        """
        Computes the (transpose) Jacobian of the propensity vector.

        Dimension of the output: (nb of parameters, nb of reactions).
        """
        propensity_first_order_derivative = prop_lib.PropensityFirstOrderDerivative(
            self.num_species, self.reactant_matrix, self.parameter_dict
        )

        param_names = list(self.parameter_dict.keys())
        propensity_jacobian = np.zeros([len(param_names), self.num_reactions])
        # Fill the Jacobian one column at a time
        for reaction_no in range(self.num_reactions):
            reaction_type = self.reaction_dict[reaction_no][0]
            if reaction_type == "mass action":
                propensity_jacobian[:, reaction_no] = (
                    propensity_first_order_derivative.mass_action(
                        state,
                        reaction_no,
                        self.reaction_dict[reaction_no][1],
                        param_names,
                    )
                )
            elif reaction_type == "hill activation":
                propensity_jacobian[:, reaction_no] = (
                    propensity_first_order_derivative.hill_activation(
                        state,
                        self.reaction_dict[reaction_no][1],
                        self.reaction_dict[reaction_no][2],
                        self.reaction_dict[reaction_no][3],
                        self.reaction_dict[reaction_no][4],
                        self.reaction_dict[reaction_no][5],
                        param_names,
                    )
                )
            elif reaction_type == "hill repression":
                propensity_jacobian[:, reaction_no] = (
                    propensity_first_order_derivative.hill_repression(
                        state,
                        self.reaction_dict[reaction_no][1],
                        self.reaction_dict[reaction_no][2],
                        self.reaction_dict[reaction_no][3],
                        self.reaction_dict[reaction_no][4],
                        self.reaction_dict[reaction_no][5],
                        param_names,
                    )
                )
            else:
                raise NotImplementedError
        return propensity_jacobian

    def log_propensity_first_order_derivative(self, state: np.array) -> np.array:
        """
        Computes the (transpose) Jacobian of the log-propensity vector.

        The Jacobian of the log-propensity cannot be calculated using the chain rule and other methods.
        When an element of the propensity vector is zero, this would return an error.

        Dimension of the output: same as Jacobian of the propensity vector, i.e. (nb of parameters, nb of reactions).
        """
        log_propensity_first_order_derivative = (
            prop_lib.LogPropensityFirstOrderDerivative(
                self.num_species, self.reactant_matrix, self.parameter_dict
            )
        )

        param_names = list(self.parameter_dict.keys())
        log_propensity_jacobian = np.zeros([len(param_names), self.num_reactions])
        # Fill the Jacobian one column at a time
        for reaction_no in range(self.num_reactions):
            reaction_type = self.reaction_dict[reaction_no][0]
            if reaction_type == "mass action":
                log_propensity_jacobian[:, reaction_no] = (
                    log_propensity_first_order_derivative.mass_action(
                        self.reaction_dict[reaction_no][1], param_names
                    )
                )
            elif reaction_type == "hill repression":
                log_propensity_jacobian[:, reaction_no] = (
                    log_propensity_first_order_derivative.hill_repression(
                        state,
                        self.reaction_dict[reaction_no][1],
                        self.reaction_dict[reaction_no][2],
                        self.reaction_dict[reaction_no][3],
                        self.reaction_dict[reaction_no][4],
                        self.reaction_dict[reaction_no][5],
                        param_names,
                    )
                )
            else:
                raise NotImplementedError
        return log_propensity_jacobian

    def propensity_second_order_derivative(self, state: np.array) -> np.array:
        """
        Computes the (generalised) Hessian of the propensity vector.

        Dimension of the output: (nb of reactions, nb of parameters, nb of parameters).
        """
        propensity_second_order_derivative = prop_lib.PropensitySecondOrderDerivative(
            self.num_species, self.reactant_matrix, self.parameter_dict
        )

        param_names = list(self.parameter_dict.keys())
        propensity_hessian = np.zeros(
            [self.num_reactions, len(param_names), len(param_names)]
        )
        # Fill the Hessian one slice (i.e. for one reaction) at a time
        for reaction_no in range(self.num_reactions):
            reaction_type = self.reaction_dict[reaction_no][0]
            if reaction_type == "mass action":
                propensity_hessian[reaction_no, :, :] = (
                    propensity_second_order_derivative.mass_action(param_names)
                )
            elif reaction_type == "hill repression":
                propensity_hessian[reaction_no, :, :] = (
                    propensity_second_order_derivative.hill_repression(
                        state,
                        self.reaction_dict[reaction_no][1],
                        self.reaction_dict[reaction_no][2],
                        self.reaction_dict[reaction_no][3],
                        self.reaction_dict[reaction_no][4],
                        self.reaction_dict[reaction_no][5],
                        param_names,
                    )
                )
            else:
                raise NotImplementedError
        return propensity_hessian

    def log_propensity_second_order_derivative(self, state: np.array) -> np.array:
        """
        Computes the (generalised) Hessian of the log-propensity vector.

        Dimension of the output: (nb of reactions, nb of parameters, nb of parameters).
        """

        log_propensity_second_order_derivative = (
            prop_lib.LogPropensitySecondOrderDerivative(
                self.num_species, self.reactant_matrix, self.parameter_dict
            )
        )

        param_names = list(self.parameter_dict.keys())
        log_propensity_hessian = np.zeros(
            [self.num_reactions, len(param_names), len(param_names)]
        )
        for reaction_no in range(self.num_reactions):
            reaction_type = self.reaction_dict[reaction_no][0]
            if reaction_type == "mass action":
                log_propensity_hessian[reaction_no, :, :] = (
                    log_propensity_second_order_derivative.mass_action(
                        self.reaction_dict[reaction_no][1], param_names
                    )
                )
            elif reaction_type == "hill repression":
                log_propensity_hessian[reaction_no, :, :] = (
                    log_propensity_second_order_derivative.hill_repression(
                        state,
                        self.reaction_dict[reaction_no][1],
                        self.reaction_dict[reaction_no][2],
                        self.reaction_dict[reaction_no][3],
                        self.reaction_dict[reaction_no][4],
                        self.reaction_dict[reaction_no][5],
                        param_names,
                    )
                )
            else:
                raise NotImplementedError
        return log_propensity_hessian

    def next_reaction_ssa(
        self,
        state: np.array,
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        """Determines the next interjump time and reaction to fire for SSA."""
        # Compute the propensities
        prop = self.propensity_vector(state)
        # Find the next interjump time and next reaction
        sum_prop = np.sum(prop)
        if sum_prop == 0:  # means that an absorbing state has been reached
            delta_t = math.inf
            next_reaction = -1
        else:
            prop = np.cumsum(prop / sum_prop)
            # Draw an exponential random variable for the next interjump time
            delta_t = -math.log(rng.uniform(0, 1)) / sum_prop
            # Choose the reaction which fired
            next_reaction = sum(prop < rng.uniform(0, 1))
        return delta_t, next_reaction

    def update_state(self, state: np.array, next_reaction: int) -> np.array:
        """Updates the state of process."""
        if next_reaction != -1:  # in case method is used although delta_t = math.inf
            # Below, be careful that numpy arrays are mutable
            state = state + self.stoichiometry_matrix[:, next_reaction]
        return state

    def output_function(self, state: np.array) -> np.array:
        """
        Computes the (whole) output function of X(t)

        Raises a NotImplementedError if not defined in the child class.
        """
        raise NotImplementedError

    ### BPA for second-order sensitivities

    def next_reaction_coupled_nrm(
        self,
        x: np.array,
        tilde_x: np.array,
        uc_i: np.array,
        tau: np.array,
        rng: np.random.Generator,
    ) -> tuple[float, int, int, np.array]:
        """Determine the next interjump time and reaction to fire for two coupled processes.

        Dimension of the output: (1,), (1,) (1,) and (nb of reactions,).
        """
        # Compute the propensities
        prop1 = self.propensity_vector(x)
        prop2 = self.propensity_vector(tilde_x)
        prop = np.zeros([self.num_reactions, 3], dtype="float64")
        prop[:, 0] = np.minimum(prop1, prop2)  # R_{k, (1,1)}, the shared process
        prop[:, 1] = prop1 - prop[:, 0]  # R_{k, (1,0)}
        prop[:, 2] = prop2 - prop[:, 0]  # R_{k, (0,1)}
        # Find the next interjump time for each reaction
        delta_t_reac_num = tau - uc_i
        delta_t_reac = np.divide(
            delta_t_reac_num,
            prop,
            out=np.full_like(delta_t_reac_num, np.inf),
            where=prop > 0,
        )
        # Retrieve the coordinates (k,b) of the minimum in the matrix delta_t_reac
        index = np.unravel_index(np.argmin(delta_t_reac, axis=None), delta_t_reac.shape)
        # Compute the next interjump time
        delta_t = delta_t_reac[index]
        # Update the internal times
        if np.sum(prop) != 0:
            uc_i += prop * delta_t
        k_star = index[0]
        l_star = index[1]
        tau[k_star, l_star] += -np.log(rng.uniform(0, 1))
        return delta_t, k_star, l_star, uc_i, tau

    def update_coupled_state(
        self, x, tilde_x, k_star, l_star
    ) -> tuple[np.array, np.array]:
        # Update $X^{1}(t)$ if $R_{k,\ell^{*}}$ with $\ell^{*}$ = (1,1) or (1,0) fired
        if l_star == 0 or l_star == 1:
            x = self.update_state(x, k_star)
        # Update $X^{2}(t)$ if $R_{k,\ell^{*}}$ with $\ell^{*}$ = (1,1) or (0,1) fired
        if l_star == 0 or l_star == 2:
            tilde_x = self.update_state(tilde_x, k_star)
        return x, tilde_x

    def generate_coupled_nrm(
        self,
        x: np.array,
        tilde_x: np.array,
        uc_t: float,
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array]:
        """Generate a pair of coupled processes.

        Dimension of the output: (nb of species,) and (nb of species,)
        """
        lc_t = 0.0
        uc_m = self.num_reactions  # nb of reactions
        # Initialize the coupled mNRM
        uc_i = np.zeros([uc_m, 3])
        tau = -np.log(rng.uniform(0, 1, [uc_m, 3]))
        while 1:
            delta_t, k_star, l_star, uc_i, tau = self.next_reaction_coupled_nrm(
                x, tilde_x, uc_i, tau, rng
            )
            lc_t += delta_t
            if lc_t > uc_t:
                return x, tilde_x
            else:
                x, tilde_x = self.update_coupled_state(x, tilde_x, k_star, l_star)

    def double_integral_difference(
        self,
        x_2: np.array,
        tilde_x_2: np.array,
        lc_t_2: float,
        delta_t_2: float,
        lc_t_0: float,
        delta_t_0: float,
        uc_t_0: float,
        u_min: float,
        u_max: float,
        domain: str,
    ) -> np.array:
        """Update the integral over part of a second-order trajectory in eq. (21).

        No trajectory is being simulated here.

        Dimension of the output: (nb of output functions,).
        """
        output_pkxql1 = self.output_function(np.expand_dims(x_2, axis=0))
        output_pkxql2 = self.output_function(np.expand_dims(tilde_x_2, axis=0))
        f_d = output_pkxql2 - output_pkxql1
        if domain == "upper triangular":
            u_star_r = uc_t_0 - (lc_t_0 + delta_t_0) - lc_t_2
            u_star_l = uc_t_0 - (lc_t_0 + delta_t_0) - (lc_t_2 + delta_t_2)
            uc_u = u_max - u_star_r
            uc_u += (u_star_r - u_star_l) / 2
        elif domain == "rectangular":
            uc_u = u_max - u_min
        elif domain == "diamond":
            u_star_r_b = uc_t_0 - (lc_t_0 + delta_t_0) - lc_t_2
            u_star_l_b = uc_t_0 - (lc_t_0 + delta_t_0) - (lc_t_2 + delta_t_2)
            u_star_r_t = uc_t_0 - lc_t_0 - lc_t_2
            u_star_l_t = uc_t_0 - lc_t_0 - (lc_t_2 + delta_t_2)
            uc_u = (u_star_r_b - u_star_l_b) / 2
            uc_u += (u_star_r_t - u_star_l_t) / 2
            uc_u += u_star_l_t - u_star_r_b
        elif domain == "lower triangular":
            u_star_r = uc_t_0 - lc_t_0 - lc_t_2
            u_star_l = uc_t_0 - lc_t_0 - (lc_t_2 + delta_t_2)
            uc_u = u_star_l - u_min
            uc_u += (u_star_r - u_star_l) / 2
        uc_u *= f_d * delta_t_2
        return uc_u

    def second_order_difference_segment(
        self,
        x_2: np.array,
        tilde_x_2: np.array,
        r_min: float,
        phi_2: float,
        uc_t_2: float,
        uc_delta_g: np.array,
        lc_t_0: float,
        delta_t_0: float,
        uc_t_0: float,
        u_min: float,
        u_max: float,
        domain: str,
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array, np.array]:
        """Compute the integral over part of a second-order trajectory in eq. (21)

        The coupled second-order pair $(X^{p,k,1,l,q,1}(t), X^{p,k,1,l,q,2}(t))$ or $(X^{p,k,2,l,q,1}(t),$
        $X^{p,k,2,l,q,2}(t))$ is simulated with the coupled mNRM. For Double BPA, $u_min = \sigma{q}$ and
        $u_max = \sigma_{q+1} ∧ (T - \sigma_{p})$.

        Dimension of the output: (nb of species,), (nb of species,) and (nb of output functions,).
        """
        uc_m = self.num_reactions  # nb of reactions
        n = self.output_function_size  # nb of output functions
        # Initialize variable for the (generalised) Hessian
        uc_h = np.zeros([n])
        # Move second-order pair forward if required
        (x_2, tilde_x_2) = self.generate_coupled_nrm(x_2, tilde_x_2, phi_2, rng)
        # Initialize the coupled mNRM
        lc_t_2 = r_min + phi_2
        uc_i = np.zeros([uc_m, 3])
        tau = -np.log(rng.uniform(0, 1, [uc_m, 3]))
        # Run the coupled mNRM
        while 1:
            # if np.all(x_2 == tilde_x_2):
            #     return x_2, tilde_x_2, uc_delta_g
            delta_t_2, o_star, m_star, uc_i, tau = self.next_reaction_coupled_nrm(
                x_2, tilde_x_2, uc_i, tau, rng
            )
            delta_t_2 = min(delta_t_2, uc_t_2 - lc_t_2)
            uc_delta_h = self.double_integral_difference(
                x_2,
                tilde_x_2,
                lc_t_2,
                delta_t_2,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                domain,
            )
            uc_h += uc_delta_h
            # Update the time
            lc_t_2 += delta_t_2
            if lc_t_2 >= uc_t_2:
                uc_delta_g += uc_h
                return x_2, tilde_x_2, uc_delta_g
            # Update the second-order process which fired
            else:
                x_2, tilde_x_2 = self.update_coupled_state(
                    x_2, tilde_x_2, o_star, m_star
                )

    def second_order_difference(
        self,
        x_2: np.array,
        tilde_x_2: np.array,
        uc_t_1: float,
        lc_t_0: float,
        delta_t_0: float,
        uc_t_0: float,
        r_bott_left: float,
        r_top_left: float,
        r_bott_right: float,
        r_top_right: float,
        u_min: float,
        u_max: float,
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array, np.array]:
        """Compute the integral over of a second-order trajectory over the domain $[\sigma^{(p,k)}_{q},  \sigma^{(p,k)}_{q+1} \wedge (T - \sigma_{p})]\times$
        $[\sigma_{p}, \text{min}(\sigma_{p+1} \wedge T, T - u)]$ in eq. (21).

        Dimension of the output: (nb of species,), (num of species,) and (nb of output functions,).
        """
        n = self.output_function_size  # nb of output functions
        uc_delta_g = np.zeros([n])
        restart_time_1 = uc_t_0 - (lc_t_0 + delta_t_0)
        if uc_t_1 == restart_time_1:  # $\text{min} = \sigma_{p+1} \wedge T$
            phi_2 = [r_bott_right, 0.0, 0.0]
            if r_bott_left <= r_top_right:  # first case in figure 3
                r_min = [0.0, r_bott_left, r_top_right]
                uc_t_2 = [r_bott_left, r_top_right, r_top_left]
                core_shape = "rectangular"
            else:  # second case in figure 3
                r_min = [0.0, r_top_right, r_bott_left]
                uc_t_2 = [r_top_right, r_bott_left, r_top_left]
                core_shape = "diamond"
            (x_2, tilde_x_2, uc_delta_g) = self.second_order_difference_segment(
                x_2,
                tilde_x_2,
                r_min[0],
                phi_2[0],
                uc_t_2[0],
                uc_delta_g,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                "upper triangular",
                rng,
            )
            (x_2, tilde_x_2, uc_delta_g) = self.second_order_difference_segment(
                x_2,
                tilde_x_2,
                r_min[1],
                phi_2[1],
                uc_t_2[1],
                uc_delta_g,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                core_shape,
                rng,
            )
            (x_2, tilde_x_2, uc_delta_g) = self.second_order_difference_segment(
                x_2,
                tilde_x_2,
                r_min[2],
                phi_2[2],
                uc_t_2[2],
                uc_delta_g,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                "lower triangular",
                rng,
            )
        else:  # $\text{min} = T - u$
            r_min = 0.0
            phi_2 = 0.0
            uc_t_2 = r_top_right
            (x_2, tilde_x_2, uc_delta_g) = self.second_order_difference_segment(
                x_2,
                tilde_x_2,
                r_min,
                phi_2,
                uc_t_2,
                uc_delta_g,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                "rectangular",
                rng,
            )
            r_min = r_top_right
            uc_t_2 = r_top_left
            (x_2, tilde_x_2, uc_delta_g) = self.second_order_difference_segment(
                x_2,
                tilde_x_2,
                r_min,
                phi_2,
                uc_t_2,
                uc_delta_g,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                u_min,
                u_max,
                "lower triangular",
                rng,
            )
        return x_2, tilde_x_2, uc_delta_g

    def first_term_difference(
        self, x_1: np.array, tilde_x_1: np.array, delta_t_1: float
    ) -> np.array:
        """Update the integral over a first-order trajectory in eq. (12).

        No trajectory is being simulated here.

        Dimension of the output: (nb of output functions,).
        """
        output_pk1 = self.output_function(np.expand_dims(x_1, axis=0))
        output_pk2 = self.output_function(np.expand_dims(tilde_x_1, axis=0))
        f_d = output_pk2 - output_pk1
        delta_uc_f = f_d * delta_t_1
        return delta_uc_f

    def first_order_difference_segment(
        self,
        x_1: np.array,
        tilde_x_1: np.array,
        phi_1: float,
        uc_t_1: float,
        uc_delta_d: np.array,
        uc_delta_e: np.array,
        lc_t_0: float,
        delta_t_0: float,
        uc_t_0: float,
        c_2: float,
        nb_sec_ap: int,
        hat_c_num: list[float, float],
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array, np.array, np.array, float]:
        """Compute the integral over of a first-order trajectory either from $0$ to $T - (\sigma_{p+1} ∧ T)$ in eq. (21)
        or from $T - (\sigma_{p+1} ∧ T)$ and $T - \sigma_{p}$ in eq. (12) and (21).

        The coupled first-order pair of processes $(X^{p,k,1}(t), X^{p,k,2}(t))$ is simulated with the coupled mNRM.

        Dimension of the output: (nb of species,), (nb of species,), (nb of output functions,),
        (nb of output functions, nb of parameters) and (1,).
        """
        lc_d = len(self.parameter_dict)  # lower case d: nb of parameters
        uc_m = self.num_reactions  # nb of reactions
        n = self.output_function_size  # nb of output functions
        # Initialize variable for the Jacobian
        uc_f = np.zeros(n)
        # Initialize variable for the (generalised) Hessian
        uc_g = np.zeros([2, n, lc_d, uc_m], dtype="float64")
        # Initialize the coupled mNRM
        lc_t_1 = phi_1
        uc_i = np.zeros([uc_m, 3])
        tau = -np.log(rng.uniform(0, 1, [uc_m, 3]))
        # Run the coupled mNRM
        while 1:
            delta_t_1, k_star, l_star, uc_i, tau = self.next_reaction_coupled_nrm(
                x_1, tilde_x_1, uc_i, tau, rng
            )
            delta_t_1 = min(delta_t_1, uc_t_1 - lc_t_1)
            if uc_t_1 == uc_t_0 - lc_t_0:
                delta_uc_f = self.first_term_difference(x_1, tilde_x_1, delta_t_1)
                uc_f += delta_uc_f
            for ttt_x in range(2):
                # Initialise second-order auxiliary path
                x_2 = np.array([x_1, tilde_x_1][ttt_x])
                # Specify the shapes for integration
                u_min = lc_t_1
                u_max = lc_t_1 + delta_t_1
                r_bott_left = uc_t_0 - (lc_t_0 + delta_t_0) - lc_t_1
                r_top_left = uc_t_0 - lc_t_0 - lc_t_1
                r_bott_right = uc_t_0 - (lc_t_0 + delta_t_0) - u_max
                r_top_right = uc_t_0 - lc_t_0 - u_max
                # Compute the Jacobian at $$X^{p,k,x}(t)$ once
                jac_1 = self.propensity_first_order_derivative(x_2)
                # Update the estimate of $C_2$, see eq. (31)
                hat_c_num[1] += delta_t_0 * delta_t_1 * np.sum(np.abs(jac_1))
                # Compute the integral over second-order trajectory
                # for each reaction $\ell$
                for l in range(uc_m):
                    gamma_num = np.sum(np.abs(jac_1[:, l]))
                    gamma_num *= delta_t_0 * delta_t_1
                    gamma = gamma_num / c_2
                    gamma = min(1, gamma)
                    # Start (coupled) second-order auxiliary paths for reaction $\ell$
                    if rng.uniform(0, 1) < gamma:  # $\beta_{pkxql} = 1$
                        nb_sec_ap += 1
                        # Initialise second-order auxiliary path
                        tilde_x_2 = self.update_state(x_2, l)
                        _, _, uc_delta_g = self.second_order_difference(
                            x_2,
                            tilde_x_2,
                            uc_t_1,
                            lc_t_0,
                            delta_t_0,
                            uc_t_0,
                            r_bott_left,
                            r_top_left,
                            r_bott_right,
                            r_top_right,
                            u_min,
                            u_max,
                            rng,
                        )
                        uc_g[ttt_x, :, :, l] += (
                            np.tensordot(uc_delta_g, jac_1[:, l], axes=0) / gamma
                        )
            # Update the time
            lc_t_1 += delta_t_1
            if lc_t_1 >= uc_t_1:
                uc_g = uc_g[1, :, :, :] - uc_g[0, :, :, :]
                uc_g = np.sum(uc_g, axis=2)
                uc_delta_d += uc_f
                uc_delta_e += uc_g
                return (x_1, tilde_x_1, uc_delta_d, uc_delta_e, nb_sec_ap, hat_c_num)
            # Update the first-order process which fired
            else:
                x_1, tilde_x_1 = self.update_coupled_state(
                    x_1, tilde_x_1, k_star, l_star
                )

    def first_order_difference(
        self,
        x_1: np.array,
        tilde_x_1: np.array,
        lc_t_0: float,
        delta_t_0: float,
        uc_t_0: float,
        c_2: float,
        nb_sec_ap: int,
        hat_c_num: list[float, float],
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array, np.array, np.array, int]:
        """Compute the integral over a first-order trajectory from $T - (\sigma_{p+1} ∧ T )$ to $T - \sigma_{p}$ in eq. (12) and
        from $0$ to $T - \sigma_{p}$ in eq. (21).

        Dimension of the output: (nb of species,), (nb of species,), (nb of output functions,),
        (nb of output functions, nb of parameters) and (1,).
        """
        lc_d = len(self.parameter_dict)  # lower case d: nb of parameters
        n = self.output_function_size  # nb of output functions
        # Initialize change to integrals
        uc_delta_d = np.zeros(n)
        uc_delta_e = np.zeros([n, lc_d], dtype="float64")
        # Integrate from $0$ to $T - (\sigma_{p+1} ∧ T)$
        phi_1 = 0
        uc_t_1 = uc_t_0 - (lc_t_0 + delta_t_0)  # $T - (\sigma_{p+1} ∧ T)$
        (x_1, tilde_x_1, uc_delta_d, uc_delta_e, nb_sec_ap, hat_c_num) = (
            self.first_order_difference_segment(
                x_1,
                tilde_x_1,
                phi_1,
                uc_t_1,
                uc_delta_d,
                uc_delta_e,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                c_2,
                nb_sec_ap,
                hat_c_num,
                rng,
            )
        )
        # Integrate from $T - (\sigma_{p+1} ∧ T)$ to $T - \sigma_{p}$
        phi_1 = uc_t_0 - (lc_t_0 + delta_t_0)  # $T - (\sigma_{p+1} ∧ T)$
        uc_t_1 = uc_t_0 - lc_t_0  # $T - \sigma_{p}$
        (x_1, tilde_x_1, uc_delta_d, uc_delta_e, nb_sec_ap, hat_c_num) = (
            self.first_order_difference_segment(
                x_1,
                tilde_x_1,
                phi_1,
                uc_t_1,
                uc_delta_d,
                uc_delta_e,
                lc_t_0,
                delta_t_0,
                uc_t_0,
                c_2,
                nb_sec_ap,
                hat_c_num,
                rng,
            )
        )
        return x_1, tilde_x_1, uc_delta_d, uc_delta_e, nb_sec_ap, hat_c_num

    def generate_bpa_second_order_sample(
        self,
        uc_t_0: float,
        c: tuple[float, float],
        # derivative_order: str,
        rng: np.random.Generator,
    ) -> tuple[np.array, np.array, np.array, np.array, list[float, float]]:
        """Generate one sample with the BPA for second-order sensitivities by computing the integral over the
        trajectory of the main process from $0$ to $T$ in eq. (12) and (21).

        The main process $(X(t))$ is simulated with SSA.

        Dimension of the output: (nb of output functions,), (nb of parameters, nb of output functions),
        (nb of output functions, nb of parameters, nb of parameters) and (2,).
        """
        lc_d = len(self.parameter_dict)  # lower case d: nb of parameters
        uc_m = self.num_reactions  # nb of reactions
        n = self.output_function_size  # nb of output functions
        # Initialize variable for the Jacobian
        uc_d_jac = np.zeros([n, lc_d, uc_m])
        # Initialize variables for the (generalised) Hessian
        uc_d_hess = np.zeros([n, lc_d, lc_d, uc_m])
        uc_e_hess = np.zeros([n, lc_d, lc_d, uc_m])
        # Set up and track auxiliary paths
        hat_c_num = [0.0, 0.0]  # to estimate $C_1$ and $C_2$
        c_1, c_2 = c
        nb_first_ap = 0  # nb of first-order processes simulated
        nb_sec_ap = 0  # nb of second-order processes simulated
        # Initialize the time and state of the main process
        lc_t_0 = 0.0  # lower case t: time of $(X(t))$
        x_0 = self.initial_state  # deterministic initial value
        # Run the SSA from $0$ to $T$
        while lc_t_0 < uc_t_0:
            delta_t_0, k_star = self.next_reaction_ssa(x_0, rng)
            # Compute $(\sigma_{p+1} ∧ T) - \sigma_{p}$
            delta_t_0 = min(delta_t_0, uc_t_0 - lc_t_0)
            # Compute the Jacobian and Hessian at $X(\sigma_{p})$ once
            jac_0 = self.propensity_first_order_derivative(x_0)
            # if derivative_order == "second":
            hess_0 = self.propensity_second_order_derivative(x_0)
            # Update the estimate of $C_1$, see eq. (28)
            hat_c_num[0] += delta_t_0 * 2 * np.sum(np.abs(jac_0))
            # if derivative_order == "second":
            hat_c_num[0] += delta_t_0 * np.sum(np.abs(hess_0))
            # Compute the integral over first-order trajectory
            # for each reaction $k$ from $0$ to $T - \sigma_{p}$
            for k in range(uc_m):
                # Compute $\gamma_{pk}$
                gamma_num = 2 * np.sum(np.abs(jac_0[:, k]))
                # if derivative_order == "second":
                gamma_num += np.sum(np.abs(hess_0[k, :, :]))
                gamma_num *= delta_t_0
                gamma = gamma_num / c_1
                gamma = min(1, gamma)
                # Start (coupled) first-order auxiliary paths for reaction $k$
                if rng.uniform(0, 1) < gamma:  # $\beta_{pk} = 1$
                    nb_first_ap += 1
                    # Initialise first-order auxiliary paths
                    x_1 = np.array(x_0)  # $ X^{p,k,1}(t)$
                    tilde_x_1 = self.update_state(x_0, k)  # $ X^{p,k,2}(t)$
                    _, _, uc_delta_d, uc_delta_e, nb_sec_ap, hat_c_num = (
                        self.first_order_difference(
                            x_1,
                            tilde_x_1,
                            lc_t_0,
                            delta_t_0,
                            uc_t_0,
                            c_2,
                            nb_sec_ap,
                            hat_c_num,
                            rng,
                        )
                    )
                    # Update BPA for first-order sensitivities for reaction $k$ and jump $\sigma_{p}$
                    uc_d_jac[:, :, k] += (
                        np.tensordot(uc_delta_d, jac_0[:, k], axes=0) / gamma
                    )
                    # if derivative_order == "second":
                    # Update first term of BPA for second-order sensitivities for reaction $k$ for jump $\sigma_{p}$
                    uc_d_hess[:, :, :, k] += (
                        np.tensordot(uc_delta_d, hess_0[k, :, :], axes=0) / gamma
                    )
                    # Update second term of BPA for second-order sensivitivities for reaction $k$ for jump $\sigma_{p}$
                    uc_e_hess[:, :, :, k] += (
                        np.tensordot(uc_delta_e, jac_0[:, k], axes=0) / gamma
                    )
            # Update the time
            lc_t_0 += delta_t_0
            # Update the main process
            if lc_t_0 < uc_t_0:  # necessary to get correct final state
                x_0 = self.update_state(x_0, k_star)
        # Compute the output
        output = self.output_function(np.expand_dims(x_0, axis=0))
        # Compute the Jacobian
        jacobian = np.sum(uc_d_jac, axis=2)
        # Compute the (generalised) Hessian
        cal_h_1 = np.sum(uc_d_hess, axis=3)
        cal_h_2 = np.sum(uc_e_hess, axis=3)
        cal_h_2 = cal_h_2 + cal_h_2.transpose((0, 2, 1))
        hessian = cal_h_1 + cal_h_2
        # Gather the number of first- and second-order APs
        nb_ap = np.array([nb_first_ap, nb_sec_ap])
        return output, jacobian, hessian, nb_ap, hat_c_num

    def estimate_normalisation_constants(
        self,
        uc_t_0: float,
        mu: list[int, int],
        num_presamples: int,
        interactive_mode_config: dict,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Estimate the normalisation constants for the Bernoulli random variables.

        Dimension of the output: (2,).
        """
        disable_progress = interactive_mode_config["disable_progress_update"]
        progress_bool = disable_progress == "True"
        # Create the variable to estimate the normalisation constants
        hat_c = [0.0, 0.0]
        # Estimate $C_1$
        print("Generating presamples to estimate C_1...")
        num_ap_tot = np.zeros(2)
        hat_c_num_tot = [0.0, 0.0]
        tmp_c = np.array([np.inf, np.inf])
        for _ in tqdm(range(num_presamples), disable=progress_bool):
            _, _, _, num_ap, hat_c_num = self.generate_bpa_second_order_sample(
                uc_t_0, tmp_c, rng
            )
            num_ap_tot += num_ap
            hat_c_num_tot[0] += hat_c_num[0]
        av_num_ap = num_ap_tot / num_presamples
        # Nb of APs should be zero
        print(
            f"Average number of pairs of APs per trajectory when estimating C_1: {av_num_ap}"
        )
        # See eq. (28)
        hat_c[0] = hat_c_num_tot[0] / (num_presamples * mu[0])
        print(f"Estimated C_1: {hat_c[0]}\n")
        # Estimate $C_2$
        print("Generating presamples to estimate C_2...")
        num_ap_tot = np.zeros(2)
        hat_c_num_tot = [0.0, 0.0]
        tmp_c[0] = hat_c[0]
        for _ in tqdm(range(num_presamples), disable=progress_bool):
            _, _, _, num_ap, hat_c_num = self.generate_bpa_second_order_sample(
                uc_t_0, tmp_c, rng
            )
            num_ap_tot += num_ap
            hat_c_num_tot[1] += hat_c_num[1]
        av_num_ap = num_ap_tot / num_presamples
        # Nb of second-order APs should be zero
        print(
            f"Average number of pairs of APs per trajectory when estimating C_2: {av_num_ap}"
        )
        # See eq. (31)
        hat_c[1] = hat_c_num_tot[1] / (num_presamples * mu[1])
        print(f"Estimated C_2: {hat_c[1]}\n")
        return hat_c

    def generate_bpa_second_order_samples(
        self,
        uc_t_0: float,
        num_samples: int,
        mu: list[int, int],
        num_presamples: int,
        interactive_mode_config: dict,
        rng: np.random.Generator,
    ) -> dict[np.array, np.array]:
        """Generate num_samples samples of the (generalised) Hessian of the expected output w.r.t. all system parameters
        with the BPA for second-order sensitivities. Samples for the mean and Jacobian are generated at the same time.
        """
        disable_progress = interactive_mode_config["disable_progress_update"]
        progress_bool = disable_progress == "True"
        lc_d = len(self.parameter_dict)  # lower case d: nb of parameters
        n = self.output_function_size  # nb of output functions
        # Find the normalisation constants for the Bernoulli random variables
        c = self.estimate_normalisation_constants(
            uc_t_0, mu, num_presamples, interactive_mode_config, rng
        )
        # Create the variables to store the samples for the mean, Jacobian and Hessian
        output_values = np.zeros([num_samples, n])
        jacobian_values = np.zeros([num_samples, n, lc_d])
        hessian_values = np.zeros([num_samples, n, lc_d, lc_d])
        # Create the variable to store the duration of each simulation
        durations = np.zeros(num_samples)
        # Create the variable to store the number of first- and second-order APs
        depth_ap_hierarchy = 2
        tot_num_ap = np.zeros(depth_ap_hierarchy)
        # Generate samples with the BPA for second-order sensitivities
        print("Generating samples with the BPA for second-order sensitivities...")
        for i in tqdm(range(num_samples), disable=progress_bool):
            start = time.time()
            output, jacobian, hessian, num_ap, _ = (
                self.generate_bpa_second_order_sample(uc_t_0, c, rng)
            )
            # Store the sample values
            output_values[i, :] = output
            jacobian_values[i, :, :] = jacobian
            hessian_values[i, :, :, :] = hessian
            # Store the duration of the simulation
            durations[i] = time.time() - start
            # Update the number of first- and second-order APs
            tot_num_ap += num_ap
        print(
            f"Number of pairs of APs per trajectory when using C_1 and C_2: {tot_num_ap / num_samples}\n"
        )
        print("Results...")
        with np.printoptions(precision=5, suppress=True):
            print(f"Estimated outputs:\n{np.mean(output_values, axis=0)}\n")
            print(f"Estimated Jacobian:\n{np.mean(jacobian_values, axis=0)}\n")
            print(f"Estimated Hessian:\n{np.mean(hessian_values, axis=0)}\n")
        results = {
            "samples": hessian_values,
            "jacobian_samples": jacobian_values,
            "output_samples": output_values,
            "simulation_times": durations,
        }
        return results

    ### Girsanov transform method for second-order sensitivities

    def generate_second_order_girsanov_sample_with_ssa(
        self, final_time: float, rng: np.random.Generator
    ) -> np.array:
        """Creates one sample for the hessian using Girsanov's method."""

        num_param = len(self.parameter_dict)
        log_prob_first_order_derivative = np.zeros([num_param, self.num_reactions])
        log_prob_second_order_derivative = np.zeros(
            [self.num_reactions, num_param, num_param]
        )
        # Initialize the time and state
        time_curr = 0
        state_curr = self.initial_state  # deterministic value
        # Run the SSA from $0$ to $T$
        while 1:
            delta_t, next_reaction = self.next_reaction_ssa(state_curr, rng)
            delta_t = min(delta_t, final_time - time_curr)
            # Update the first-order derivative of likelihood
            prop_jacobian = self.propensity_first_order_derivative(state_curr)
            log_prob_first_order_derivative -= prop_jacobian * delta_t
            # Update the second-order derivative of likelihood
            prop_hessian = self.propensity_second_order_derivative(state_curr)
            log_prob_second_order_derivative -= prop_hessian * delta_t
            time_curr += delta_t
            # Update the time
            if time_curr < final_time:
                one_hot_vector = np.zeros(self.num_reactions)
                one_hot_vector[next_reaction] = 1
                # Update the first-order derivative of the log-likelihood
                log_jacobian = self.log_propensity_first_order_derivative(state_curr)
                log_prob_first_order_derivative += log_jacobian * one_hot_vector
                # Update the second-order derivative of the log-likelihood
                log_hessian = self.log_propensity_second_order_derivative(state_curr)
                log_prob_second_order_derivative += (
                    log_hessian * one_hot_vector.reshape((self.num_reactions, 1, 1))
                )
                # Update $X(t)$
                state_curr = self.update_state(state_curr, next_reaction)
            else:
                # Compute the first-order derivative of the log-likelihood
                log_prob_first_order_derivative = np.sum(
                    log_prob_first_order_derivative, axis=1
                )
                # Compute the second-order derivative of the log-likelihood
                log_prob_second_order_derivative = np.sum(
                    log_prob_second_order_derivative, axis=0
                )
                # Compute the output
                output = self.output_function(np.expand_dims(state_curr, axis=0))
                # Compute the Hessian
                hessian = (
                    np.outer(
                        log_prob_first_order_derivative,
                        log_prob_first_order_derivative,
                    )
                    + log_prob_second_order_derivative
                ) * self.output_function(np.expand_dims(state_curr, axis=0)).reshape(
                    (self.output_function_size, 1, 1)
                )
                return output, hessian

    def generate_second_order_girsanov_samples_with_ssa(
        self,
        final_time,
        num_samples,
        interactive_mode_config: dict,
        rng: np.random.Generator,
    ):
        disable_progress = interactive_mode_config["disable_progress_update"]
        progress_bool = disable_progress == "True"
        # Create the variables to store the samples for the mean and and Hessian
        num_param = len(self.parameter_dict)
        output_values = np.zeros([num_samples, self.output_function_size])
        hessian_values = np.zeros(
            [num_samples, self.output_function_size, num_param, num_param]
        )
        # Create the variable to store the duration of each simulation
        durations = np.zeros(num_samples)
        # Generate the second-order Girsanov samples
        print(
            "Generating samples with the Girsanov transform for second-order sensitivities..."
        )
        for i in tqdm(range(num_samples), disable=progress_bool):
            start = time.time()
            # hessian_values[i, :, :, :] = (
            #     self.generate_second_order_girsanov_sample_with_ssa(final_time, rng)
            # )
            output, sample_value = self.generate_second_order_girsanov_sample_with_ssa(
                final_time, rng
            )
            # Store the sample values
            output_values[i, :] = output
            hessian_values[i, :, :, :] = sample_value
            # Store the duration of the simulation
            durations[i] = time.time() - start
        print("\nResults...")
        with np.printoptions(precision=5, suppress=True):
            print(f"Estimated outputs:\n{np.mean(output_values, axis=0)}\n")
            print(f"Estimated Hessian:\n{np.mean(hessian_values, axis=0)}\n")
        results = {
            "samples": hessian_values,
            "simulation_times": np.array(durations),
        }
        return results
