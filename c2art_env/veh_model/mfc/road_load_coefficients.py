def compute_f_coefficients(powertrain, width, height, veh_mass, rolling_coef, aero_coef, res_coef_1, res_coef_2, res_coef_3):
    f0 = veh_mass * rolling_coef * 9.80665
    f2 = 0.5 * 1.2 * (res_coef_1 * width * height * aero_coef) / pow(3.6, 2)
    if powertrain == 'electric engine':
        f1 = 0
    else:
        f1 = res_coef_2 * f2 + res_coef_3
    return f0, f1, f2
