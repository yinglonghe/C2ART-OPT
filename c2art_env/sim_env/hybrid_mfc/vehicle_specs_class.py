'''Define veh_specs class'''


class veh_specs(object):

    # The class "constructor"
    def __init__(self, my_car, powertrain, **kwargs):
        '''
        kwargs can be:
        lco = True          # Light co2mpas is to be used, so the relevant parameters must be imported
        electric = True     # The vehicle is an EV

        :param my_car:
        :param kwargs:
        '''

        if powertrain == 'conventional':
            self.engine_max_power = float(my_car["Fuel Engine-Max power"])  # kW
            self.engine_max_speed_at_max_power = float(my_car["Fuel Engine-Max power RPM"])  # rpm
            gr_str = my_car["Transmission  / Gear ratio-Gear Box Ratios"]  # [3.33, 1.95, 1.29, 0.98, 0.76]
            gr_str = gr_str[1:-1].split('-')
            gr = []
            for i in range(len(gr_str)):
                gr.append(float(gr_str[i]))
            self.gr = gr
            self.fuel_type = my_car["Drive-Fuel"]
            if self.fuel_type == 'petrol':
                self.ignition_type = 'positive'
            elif self.fuel_type == 'diesel':
                self.ignition_type = 'compression'
            else:
                print("neither petrol nor diesel, I exit!")
                exit()
            self.tire_radius = float(my_car["Chassis-Rolling Radius Static"]) / 1000  # meters
            self.driveline_slippage = 0.00
            if my_car["General Specifications-Transmission"] == 'automatic':
                self.driveline_efficiency = 0.90
                self.transmission = 'automatic'
            else:
                self.driveline_efficiency = 0.93
                self.transmission = 'manual'
            self.final_drive = float(my_car["Transmission  / Gear ratio-Final drive"])
            self.veh_mass = float(my_car["Weights-Empty mass"])
            self.veh_inertia_mass = float(my_car["Weights-Inertia mass"])
            self.phi = float(self.veh_inertia_mass/self.veh_mass)
            self.top_speed = float(my_car["Performance-Top speed"]) / 3.6
            self.time_0_100 = float(my_car["Performance-Acceleration 0-100 km/h"])
            self.type_of_car = my_car["General Specifications-Carbody"].strip()
            self.car_width = float(my_car["Exterior sizes-Width"])
            self.car_height = float(my_car["Exterior sizes-Height"])
            self.car_length = float(my_car["Exterior sizes-Length"])
            self.kerb_weight = float(my_car["Weights-Unladen mass"])
            self.wheelbase = float(my_car["Exterior sizes-Wheelbase"])
            self.powertrain = my_car["Drive-Drive system"]
            self.model = my_car["Vehicle-Model"]
            self.release_year = my_car["General Specifications-Release date"]
            self.url = my_car["Vehicle-Url"]
            car_type = my_car["Drive-Wheel drive"]
            if car_type == 'front':
                self.car_type = 1
            elif car_type == 'rear':
                self.car_type = 2
            else:
                self.car_type = 3
            if self.ignition_type == "positive":
                self.idle_engine_speed = (750, 50)
            else:
                self.idle_engine_speed = (850, 50)

            if 'lco' in kwargs:
                if kwargs['lco']:
                    '''Used for Light Co2mpass'''
                    self.r_dynamic = float(my_car["Chassis-Rolling Radius Dynamic"]) / 1000
                    self.engine_max_torque = float(my_car["Fuel Engine-Max torque"])
                    self.fuel_engine_stroke = float(my_car["Fuel Engine-Stroke"])
                    self.max_power = float(my_car["Drive-Total max power"])
                    self.fuel_turbo = my_car["Fuel Engine-Turbo"]
                    self.fuel_eng_capacity = float(my_car["Fuel Engine-Capacity"])
                    self.gearbox_type = str(my_car["General Specifications-Transmission"])

        elif powertrain == 'electric':
            self.fuel_type = my_car["Drive-Fuel"]
            if self.fuel_type != 'electricity':
                pass
            self.ignition_type = 'electricity'
            self.motor_max_power = float(my_car["Electric Engine-Total max power"])  # kW
            self.motor_max_torque = float(my_car["Electric Engine-Max torque"])  # Nm
            gr_str = my_car["Transmission  / Gear ratio-Gear Box Ratios"]
            gr_str = gr_str[1:-1].split('-')
            gr = []
            for i in range(len(gr_str)):
                gr.append(float(gr_str[i]))
            self.gr = gr
            self.tire_radius = float(my_car["Chassis-Rolling Radius Static"]) / 1000  # meters
            self.driveline_slippage = 0.00
            if my_car["General Specifications-Transmission"] == 'single-speed fixed gear':
                self.driveline_efficiency = 0.90
            else:
                self.driveline_efficiency = 0.93
            self.final_drive = float(my_car["Transmission  / Gear ratio-Final drive"])
            self.veh_mass = float(my_car["Weights-Empty mass"])
            self.veh_inertia_mass = float(my_car["Weights-Inertia mass"])
            self.phi = float(self.veh_inertia_mass/self.veh_mass)
            self.top_speed = float(my_car["Performance-Top speed"]) / 3.6  # m/s
            self.time_0_100 = float(my_car["Performance-Acceleration 0-100 km/h"])
            self.type_of_car = my_car["General Specifications-Carbody"].strip()
            self.car_width = float(my_car["Exterior sizes-Width"])
            self.car_height = float(my_car["Exterior sizes-Height"])
            self.car_length = float(my_car["Exterior sizes-Length"])
            # self.kerb_weight = float(my_car["Weights-Unladen mass"])
            self.wheelbase = float(my_car["Exterior sizes-Wheelbase"])
            self.powertrain = my_car["Drive-Drive system"]
            self.model = my_car["Vehicle-Model"]
            self.release_year = my_car["General Specifications-Release date"]
            self.url = my_car["Vehicle-Url"]
            car_type = my_car["Drive-Wheel drive"]
            if car_type == 'front':
                self.car_type = 1
            elif car_type == 'rear':
                self.car_type = 2
            else:
                self.car_type = 3

        elif powertrain == 'hybrid':
            self.motor_max_power = float(my_car["Electric Engine-Total max power"])  # kW
            self.motor_max_torque = float(my_car["Electric Engine-Max torque"])  # Nm
            
            self.max_power = float(my_car["Drive-Total max power"])

            self.engine_max_power = float(my_car["Fuel Engine-Max power"])  # kW
            self.engine_max_speed_at_max_power = float(my_car["Fuel Engine-Max power RPM"])  # rpm
            gr_str = my_car["Transmission  / Gear ratio-Gear Box Ratios"]
            gr_str = gr_str[1:-1].split('-')
            gr = []
            for i in range(len(gr_str)):
                gr.append(float(gr_str[i]))
            self.gr = gr
            self.fuel_type = my_car["Drive-Fuel"]
            if self.fuel_type == 'petrol':
                self.ignition_type = 'positive'
            elif self.fuel_type == 'diesel':
                self.ignition_type = 'compression'
            else:
                print("neither petrol nor diesel, I exit!")
                exit()
            self.tire_radius = float(my_car["Chassis-Rolling Radius Static"]) / 1000  # meters
            self.driveline_slippage = 0.00
            if my_car["General Specifications-Transmission"] == 'automatic':
                self.driveline_efficiency = 0.90
                self.transmission = 'automatic'
            else:
                self.driveline_efficiency = 0.93
                self.transmission = 'manual'
            self.final_drive = float(my_car["Transmission  / Gear ratio-Final drive"])
            self.veh_mass = float(my_car["Weights-Empty mass"])
            self.veh_inertia_mass = float(my_car["Weights-Inertia mass"])
            self.phi = float(self.veh_inertia_mass/self.veh_mass)
            self.top_speed = float(my_car["Performance-Top speed"]) / 3.6
            self.time_0_100 = float(my_car["Performance-Acceleration 0-100 km/h"])
            self.type_of_car = my_car["General Specifications-Carbody"].strip()
            self.car_width = float(my_car["Exterior sizes-Width"])
            self.car_height = float(my_car["Exterior sizes-Height"])
            self.car_length = float(my_car["Exterior sizes-Length"])
            self.kerb_weight = float(my_car["Weights-Unladen mass"])
            self.wheelbase = float(my_car["Exterior sizes-Wheelbase"])
            self.powertrain = my_car["Drive-Drive system"]
            self.model = my_car["Vehicle-Model"]
            self.release_year = my_car["General Specifications-Release date"]
            self.url = my_car["Vehicle-Url"]
            car_type = my_car["Drive-Wheel drive"]
            if car_type == 'front':
                self.car_type = 1
            elif car_type == 'rear':
                self.car_type = 2
            else:
                self.car_type = 3
            if self.ignition_type == "positive":
                self.idle_engine_speed = (750, 50)
            else:
                self.idle_engine_speed = (850, 50)

            if 'lco' in kwargs:
                if kwargs['lco']:
                    '''Used for Light Co2mpass'''
                    self.r_dynamic = float(my_car["Chassis-Rolling Radius Dynamic"]) / 1000
                    self.engine_max_torque = float(my_car["Fuel Engine-Max torque"])
                    self.fuel_engine_stroke = float(my_car["Fuel Engine-Stroke"])
                    self.max_power = float(my_car["Drive-Total max power"])
                    self.fuel_turbo = my_car["Fuel Engine-Turbo"]
                    self.fuel_eng_capacity = float(my_car["Fuel Engine-Capacity"])
                    self.gearbox_type = str(my_car["General Specifications-Transmission"])


class hardcoded_params(object):
    '''
    The class is used for some params that are hard coded
    '''

    # The class "constructor"
    def __init__(self):
        # PARAMETERS HARDCODED - TO BE LOADED FROM OUTSIDE
        self.final_drive_efficiency = 0.98
        self.thres = 1100
        self.min_engine_on_speed = 600

        self.params_gearbox_losses = {
            'Automatic': {
                'gbp00': {'m': -0.0043233434399999994,
                          'q': -0.29823614099999995},
                'gbp10': {'m': -2.4525999999999996e-06,
                          'q': -0.0001547871},
                'gbp01': {'q': 0.9793688500000001},
            },
            'Manual': {
                'gbp00': {'m': -0.0043233434399999994,
                          'q': -0.29823614099999995},
                'gbp10': {'m': -2.4525999999999996e-06,
                          'q': -5.15957e-05},
                'gbp01': {'q': 0.9895177500000001},
            }
        }
        # FIX It is petrol not gasoline in the car db.
        #: Fuel density [g/l].
        self.FUEL_DENSITY = {
            'petrol': 745.0,
            'diesel': 832.0,
            'LPG': 43200.0 / 46000.0 * 745.0,  # Gasoline equivalent.
            'NG': 43200.0 / 45100.0 * 745.0,  # Gasoline equivalent.
            'ethanol': 794.0,
            'biodiesel': 890.0,
        }

        self.LHV = {
            'petrol': 43200.0,
            'diesel': 43100.0,
            'LPG': 46000.0,
            'NG': 45100.0,
            'ethanol': 26800.0,
            'biodiesel': 37900.0,
        }

        self.idle_engine_speed_median = {
            'petrol': self.min_engine_on_speed + 300,
            'diesel': self.min_engine_on_speed + 200
        }