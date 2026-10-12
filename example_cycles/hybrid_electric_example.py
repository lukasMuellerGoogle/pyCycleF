import sys
import time

import openmdao.api as om
import pycycle.api as pyc

from pycycle.elements.electric_machine import ElectricMotor, ElectricGenerator
from pycycle.elements.power_electronics import PowerElectronics

class HybridTurboshaft(pyc.Cycle):
    def setup(self):
        # We will build a simple turboshaft that drives a generator, which powers a motor.

        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('compressor', pyc.Compressor(map_data=pyc.HPCMap))
        self.add_subsystem('burner', pyc.Combustor(fuel_type='FAR'))
        self.add_subsystem('turbine', pyc.Turbine(map_data=pyc.HPTMap))
        self.add_subsystem('nozzle', pyc.Nozzle())
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2))
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # Electric components
        self.add_subsystem('generator', ElectricGenerator())
        self.add_subsystem('rectifier', PowerElectronics(is_inverter=False))
        self.add_subsystem('inverter', PowerElectronics(is_inverter=True))
        self.add_subsystem('motor', ElectricMotor())

        # Connect aerodynamics
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'compressor.Fl_I')
        self.pyc_connect_flow('compressor.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turbine.Fl_I')
        self.pyc_connect_flow('turbine.Fl_O', 'nozzle.Fl_I')

        self.connect('compressor.trq', 'shaft.trq_0')
        self.connect('turbine.trq', 'shaft.trq_1')
        self.connect('shaft.Nmech', 'compressor.Nmech')
        self.connect('shaft.Nmech', 'turbine.Nmech')
        self.connect('shaft.Nmech', 'generator.Nmech')

        # Generator consumes torque from shaft, let's say it's an external load on the turboshaft
        self.add_subsystem('balance', om.BalanceComp())
        self.balance.add_balance('W', units='lbm/s', eq_units='hp', lower=1.0, upper=100.0)
        self.balance.add_balance('FAR', eq_units='degR', lower=1e-4, upper=0.03)

        self.connect('balance.W', 'fc.W')
        self.connect('balance.FAR', 'burner.Fl_I:FAR')

        # We will extract 50 kW via the generator
        self.connect('generator.P_elec', 'rectifier.P_in')
        self.connect('rectifier.P_out', 'inverter.P_in')
        self.connect('inverter.P_out', 'motor.P_elec')

        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('compressor.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')

        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['atol'] = 1e-6
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 15

        self.linear_solver = om.DirectSolver()

        super().setup()


if __name__ == "__main__":
    prob = om.Problem()
    prob.model = HybridTurboshaft()
    prob.setup()

    # Just checking it builds successfully
    print("Hybrid turboshaft example built successfully.")
