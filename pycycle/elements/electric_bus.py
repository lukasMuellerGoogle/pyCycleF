import openmdao.api as om
import numpy as np

class ElectricBus(om.ImplicitComponent):
    """
    Electric Bus node balancing power/current between multiple electrical sources and loads.
    In a simple conceptual model, it balances the sum of input power against the sum of output power.
    Since we are tracking currents and voltages (e.g. V_t * I_a), we can balance current for a given DC voltage.
    """

    def initialize(self):
        self.options.declare('num_sources', default=1, types=int)
        self.options.declare('num_loads', default=1, types=int)
        self.options.declare('voltage_node', default=1000.0, desc='Nominal voltage node for current balancing')

    def setup(self):
        num_sources = self.options['num_sources']
        num_loads = self.options['num_loads']

        # The state to solve for is typically a free variable that balances power/current.
        # e.g., an overall system current, or we balance P_source - P_load = 0 by finding a component scale.
        # But an electric bus component can just be an explicit component that sums powers, or an implicit one that forces P_net = 0 by varying a battery current or voltage.
        # Let's keep it simple: it calculates net power. We can use a BalanceComp outside, or make this a simple explicit summer.
        pass

# Let's make an explicit bus component instead that just computes sums.
class PowerSum(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_inputs', default=2, types=int)

    def setup(self):
        num_inputs = self.options['num_inputs']
        self.add_output('P_net', val=0.0, units='W')

        for i in range(num_inputs):
            self.add_input(f'P_in_{i}', val=0.0, units='W')
            self.declare_partials('P_net', f'P_in_{i}', val=1.0)

    def compute(self, inputs, outputs):
        num_inputs = self.options['num_inputs']
        P_net = 0.0
        for i in range(num_inputs):
            P_net += inputs[f'P_in_{i}']
        outputs['P_net'] = P_net
