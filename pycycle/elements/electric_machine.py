import numpy as np
import openmdao.api as om

class ElectricMachine(om.ExplicitComponent):
    """
    Electric Machine (Motor/Generator) surrogate model based on a polynomial loss build-up
    and torque-based mass scaling.
    """

    def initialize(self):
        self.options.declare('is_motor', types=bool, default=True,
                             desc="If True, behaves as a Motor. If False, behaves as a Generator.")

    def setup(self):
        # Coefficients for mass model
        self.add_input('k_0', val=0.0, units='kg', desc='Static housing mass')
        self.add_input('k_1', val=0.0, units='kg/(N*m)', desc='Linear active mass scaling')
        self.add_input('k_2', val=0.0, desc='Non-linear structural scaling coefficient')
        self.add_input('Q_nom', val=1000.0, units='N*m', desc='Nominal design torque for mass scaling')

        # Coefficients for loss model
        self.add_input('C_0', val=0.0, units='W', desc='Parasitic baseline loss')
        self.add_input('C_1', val=0.0, units='W/(rad/s)', desc='Hysteresis loss coefficient')
        self.add_input('C_2', val=0.0, units='W/(rad/s)**2', desc='Eddy current / bearing friction loss coefficient')
        self.add_input('C_3', val=0.0, units='W/(rad/s)**3', desc='Windage loss coefficient')
        self.add_input('C_4', val=0.0, units='W/(N*m)**2', desc='Copper loss coefficient')
        self.add_input('C_5', val=0.0, units='W/((N*m)**2 * (rad/s))', desc='Load-dependent stray loss coefficient')

        # State variables
        self.add_input('Nmech', val=1000.0, units='rpm', desc='Mechanical shaft speed')
        self.add_input('trq', val=1000.0, units='ft*lbf', desc='Mechanical torque')

        # Outputs
        self.add_output('mass', val=0.0, units='kg', desc='Total electric machine mass')
        self.add_output('P_loss', val=0.0, units='W', desc='Total power loss')
        self.add_output('P_mech', val=0.0, units='W', desc='Mechanical power')
        self.add_output('P_elec', val=0.0, units='W', desc='Electrical power (input for motor, output for generator)')

        # PyCycle Shaft connection requirement
        self.add_output('power', val=0.0, units='hp', desc='Mechanical power for shaft balance')

        # Setup Partials
        self.declare_partials('mass', ['k_0', 'k_1', 'k_2', 'Q_nom'])
        self.declare_partials('P_loss', ['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'Nmech', 'trq'])
        self.declare_partials('P_mech', ['Nmech', 'trq'])
        self.declare_partials('P_elec', ['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'Nmech', 'trq'])
        self.declare_partials('power', ['Nmech', 'trq'])

    def compute(self, inputs, outputs):
        is_motor = self.options['is_motor']

        k_0 = inputs['k_0']
        k_1 = inputs['k_1']
        k_2 = inputs['k_2']
        Q_nom = inputs['Q_nom']

        C_0 = inputs['C_0']
        C_1 = inputs['C_1']
        C_2 = inputs['C_2']
        C_3 = inputs['C_3']
        C_4 = inputs['C_4']
        C_5 = inputs['C_5']

        Nmech_rpm = inputs['Nmech']
        omega = Nmech_rpm * (np.pi / 30.0) # rpm to rad/s

        trq_ft_lbf = inputs['trq']
        Q = trq_ft_lbf * 1.3558179483314004 # ft-lbf to N-m

        # Mass
        outputs['mass'] = k_0 + k_1 * Q_nom + k_2 * (Q_nom ** (3.0 / 3.5))

        # Losses
        P_loss = C_0 + C_1 * omega + C_2 * (omega**2) + C_3 * (omega**3) + C_4 * (Q**2) + C_5 * (Q**2) * omega
        outputs['P_loss'] = P_loss

        # Mech Power
        P_mech = omega * Q
        outputs['P_mech'] = P_mech

        # Convert to hp for shaft connection
        outputs['power'] = P_mech / 745.699872

        # Electrical Power
        if is_motor:
            outputs['P_elec'] = P_mech + P_loss
            # Torque supplied to shaft is positive, motor provides power to shaft
            # Actually, standard pyCycle Shaft convention:
            # trq_in: power source (turbine) - positive torque means adding power to shaft
            # trq_out: power sink (compressor) - positive torque means removing power from shaft
            # Let's just output power, Shaft uses trq variables directly.
        else:
            outputs['P_elec'] = P_mech - P_loss

    def compute_partials(self, inputs, partials):
        is_motor = self.options['is_motor']

        k_1 = inputs['k_1']
        k_2 = inputs['k_2']
        Q_nom = inputs['Q_nom']

        C_1 = inputs['C_1']
        C_2 = inputs['C_2']
        C_3 = inputs['C_3']
        C_4 = inputs['C_4']
        C_5 = inputs['C_5']

        Nmech_rpm = inputs['Nmech']
        omega = Nmech_rpm * (np.pi / 30.0)
        domega_dNmech = np.pi / 30.0

        trq_ft_lbf = inputs['trq']
        Q = trq_ft_lbf * 1.3558179483314004
        dQ_dtrq = 1.3558179483314004

        # Partials of Mass
        partials['mass', 'k_0'] = 1.0
        partials['mass', 'k_1'] = Q_nom
        partials['mass', 'k_2'] = Q_nom ** (3.0 / 3.5)
        partials['mass', 'Q_nom'] = k_1 + k_2 * (3.0 / 3.5) * (Q_nom ** (3.0 / 3.5 - 1.0))

        # Partials of P_loss w.r.t coefficients
        partials['P_loss', 'C_0'] = 1.0
        partials['P_loss', 'C_1'] = omega
        partials['P_loss', 'C_2'] = omega**2
        partials['P_loss', 'C_3'] = omega**3
        partials['P_loss', 'C_4'] = Q**2
        partials['P_loss', 'C_5'] = (Q**2) * omega

        # Partials of P_loss w.r.t state variables
        dP_loss_domega = C_1 + 2 * C_2 * omega + 3 * C_3 * (omega**2) + C_5 * (Q**2)
        partials['P_loss', 'Nmech'] = dP_loss_domega * domega_dNmech

        dP_loss_dQ = 2 * C_4 * Q + 2 * C_5 * Q * omega
        partials['P_loss', 'trq'] = dP_loss_dQ * dQ_dtrq

        # Partials of P_mech
        dP_mech_domega = Q
        dP_mech_dQ = omega
        partials['P_mech', 'Nmech'] = dP_mech_domega * domega_dNmech
        partials['P_mech', 'trq'] = dP_mech_dQ * dQ_dtrq

        partials['power', 'Nmech'] = partials['P_mech', 'Nmech'] / 745.699872
        partials['power', 'trq'] = partials['P_mech', 'trq'] / 745.699872

        # Partials of P_elec
        for var in ['C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5']:
            if is_motor:
                partials['P_elec', var] = partials['P_loss', var]
            else:
                partials['P_elec', var] = -partials['P_loss', var]

        if is_motor:
            partials['P_elec', 'Nmech'] = partials['P_mech', 'Nmech'] + partials['P_loss', 'Nmech']
            partials['P_elec', 'trq'] = partials['P_mech', 'trq'] + partials['P_loss', 'trq']
        else:
            partials['P_elec', 'Nmech'] = partials['P_mech', 'Nmech'] - partials['P_loss', 'Nmech']
            partials['P_elec', 'trq'] = partials['P_mech', 'trq'] - partials['P_loss', 'trq']

class ElectricMotor(ElectricMachine):
    def initialize(self):
        super().initialize()
        self.options['is_motor'] = True

class ElectricGenerator(ElectricMachine):
    def initialize(self):
        super().initialize()
        self.options['is_motor'] = False
