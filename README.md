# DoubleBPA

This is the code accompanying the manuscript titled _Unbiased estimation of second-order parameter sensitivities for stochastic reaction networks_ by Quentin Badolle, Ankit Gupta and Mustafa Khammash.

## Command Line Execution Example

```python
import numpy as np
import reaction_network_library as rxn_examples

crn_class_name = "ConstitutiveGeneExpression"
network = getattr(rxn_examples, crn_class_name)()

final_time = 1.0
num_samples = 10**4
num_auxiliary_paths_desired = [10**1, 10**2]
num_presamples = 100
interactive_mode_config = {"disable_progress_update": "False","simulation_progress_interval": 10.0}
rng = np.random.default_rng(seed=1)
results = network.generate_bpa_second_order_samples(
    final_time,
    num_samples,
    num_auxiliary_paths_desired,
    num_presamples,
    interactive_mode_config,
    rng,
)
```
