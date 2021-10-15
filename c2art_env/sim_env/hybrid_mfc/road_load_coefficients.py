def compute_f_coefficients(powertrain, type_of_car, width, height, veh_mass):
    # Empty dict
    d = {}
    # Fill in the entries one by one
    d["cabriolet"] = 0.28
    d["sedan"] = 0.27
    d["hatchback"] = 0.3
    d["stationwagon"] = 0.28
    d["suv/crossover"] = 0.35
    d["mpv"] = 0.3
    d["coupe"] = 0.27
    d["pick-up"] = 0.4

    rolling_res_coef = 0.009  # Constant for the moment
    theor_aero_coeff = d[type_of_car]

    f0 = veh_mass * rolling_res_coef * 9.80665
    f2 = 0.5 * 1.2 * (0.84 * width * height * theor_aero_coeff) / pow(3.6, 2)
    if powertrain == 'electric engine':
        f1 = 0
    else:
        f1 = -71.735 * f2 + 2.7609

    return f0, f1, f2
