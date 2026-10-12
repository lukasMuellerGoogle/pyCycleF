import openmdao.api as om

class PowerElectronics(om.ExplicitComponent):
    """
    Power Electronics (Inverter/Rectifier) surrogate model.
    Models conduction and switching losses based on WBG equations.
    """

    def initialize(self):
        self.options.declare('is_inverter', types=bool, default=True,
                             desc="If True, acts as an inverter (DC to AC). If False, acts as rectifier (AC to DC).")

    def setup(self):
        self.add_input('I_rms', val=100.0, units='A', desc='RMS phase current')
        self.add_input('V_dc', val=1000.0, units='V', desc='DC link voltage')
        self.add_input('f_sw', val=20000.0, units='Hz', desc='Switching frequency')

        # Conduction constants
        self.add_input('N_sw', val=6.0, desc='Number of active switches')
        self.add_input('R_ds_on', val=0.005, units='ohm', desc='On-state resistance of switches')

        # Switching constants (Energy per pulse)
        self.add_input('E_on', val=0.001, units='J', desc='Turn-on energy per pulse')
        self.add_input('E_off', val=0.001, units='J', desc='Turn-off energy per pulse')
        self.add_input('E_rec', val=0.0005, units='J', desc='Diode reverse recovery energy per pulse')

        # Power flowing through
        self.add_input('P_in', val=100000.0, units='W', desc='Input power to the electronics')

        self.add_output('P_cond', val=0.0, units='W', desc='Conduction losses')
        self.add_output('P_sw', val=0.0, units='W', desc='Switching losses')
        self.add_output('P_loss', val=0.0, units='W', desc='Total power electronic loss')
        self.add_output('P_out', val=0.0, units='W', desc='Output power from the electronics')

        self.declare_partials('P_cond', ['N_sw', 'I_rms', 'R_ds_on'])
        self.declare_partials('P_sw', ['f_sw', 'E_on', 'E_off', 'E_rec'])
        self.declare_partials('P_loss', ['N_sw', 'I_rms', 'R_ds_on', 'f_sw', 'E_on', 'E_off', 'E_rec'])
        self.declare_partials('P_out', ['P_in', 'N_sw', 'I_rms', 'R_ds_on', 'f_sw', 'E_on', 'E_off', 'E_rec'])

    def compute(self, inputs, outputs):
        I_rms = inputs['I_rms']
        N_sw = inputs['N_sw']
        R_ds_on = inputs['R_ds_on']

        f_sw = inputs['f_sw']
        E_on = inputs['E_on']
        E_off = inputs['E_off']
        E_rec = inputs['E_rec']

        P_in = inputs['P_in']

        P_cond = N_sw * (I_rms**2) * R_ds_on
        outputs['P_cond'] = P_cond

        P_sw = f_sw * (E_on + E_off + E_rec)
        outputs['P_sw'] = P_sw

        P_loss = P_cond + P_sw
        outputs['P_loss'] = P_loss

        outputs['P_out'] = P_in - P_loss

    def compute_partials(self, inputs, partials):
        I_rms = inputs['I_rms']
        N_sw = inputs['N_sw']
        R_ds_on = inputs['R_ds_on']

        f_sw = inputs['f_sw']
        E_on = inputs['E_on']
        E_off = inputs['E_off']
        E_rec = inputs['E_rec']

        partials['P_cond', 'N_sw'] = (I_rms**2) * R_ds_on
        partials['P_cond', 'I_rms'] = N_sw * 2 * I_rms * R_ds_on
        partials['P_cond', 'R_ds_on'] = N_sw * (I_rms**2)

        partials['P_sw', 'f_sw'] = E_on + E_off + E_rec
        partials['P_sw', 'E_on'] = f_sw
        partials['P_sw', 'E_off'] = f_sw
        partials['P_sw', 'E_rec'] = f_sw

        partials['P_loss', 'N_sw'] = partials['P_cond', 'N_sw']
        partials['P_loss', 'I_rms'] = partials['P_cond', 'I_rms']
        partials['P_loss', 'R_ds_on'] = partials['P_cond', 'R_ds_on']

        partials['P_loss', 'f_sw'] = partials['P_sw', 'f_sw']
        partials['P_loss', 'E_on'] = partials['P_sw', 'E_on']
        partials['P_loss', 'E_off'] = partials['P_sw', 'E_off']
        partials['P_loss', 'E_rec'] = partials['P_sw', 'E_rec']

        partials['P_out', 'P_in'] = 1.0

        for var in ['N_sw', 'I_rms', 'R_ds_on', 'f_sw', 'E_on', 'E_off', 'E_rec']:
            partials['P_out', var] = -partials['P_loss', var]
