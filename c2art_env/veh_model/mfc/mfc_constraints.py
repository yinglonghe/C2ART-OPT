import os
import numpy as np
import c2art_env.veh_model.mfc.mfc_acc as driver
import c2art_env.veh_model.mfc.reading_n_organizing as rno
from c2art_env.veh_model.mfc.road_load_coefficients import compute_f_coefficients


def mfc_curves(
        car_id,
        veh_load,
        rolling_coef,
        aero_coef,
        res_coef_1,
        res_coef_2,
        res_coef_3,
        ppar0,
        ppar1,
        ppar2,
        mh_base,
        **kwargs):

    db_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'car_database', '2019_07_03_car_db')
    db = rno.load_db_to_dictionary(db_name)
    car = rno.get_vehicle_from_db(db, car_id)

    veh_mass = car.veh_mass + veh_load

    f0, f1, f2 = compute_f_coefficients(
        car.powertrain,
        car.car_width,
        car.car_height,
        veh_mass,
        rolling_coef,
        aero_coef,
        res_coef_1,
        res_coef_2,
        res_coef_3
    )

    veh_max_speed = int(car.top_speed)

    # Conventional vehicle simulation module
    if car.powertrain == 'fuel engine':

        curves = driver.gear_curves(
            car,
            veh_mass,
            f0,
            f1,
            f2,
            mh_base
        )

        veh_model_speed = list(np.arange(0, veh_max_speed + 0.1, 0.1))  # m/s
        veh_model_acc = []
        veh_model_dec = []
        ppar = [ppar0, ppar1, ppar2]
        dec_curves = np.poly1d(ppar)

        for k in range(len(veh_model_speed)):
            acc_temp = []
            for i in range(len(curves[0])):
                acc_temp.append(float(curves[0][i](veh_model_speed[k])))
            veh_model_acc.append(max(max(acc_temp), 0.5))
            veh_model_dec.append(min(dec_curves(veh_model_speed[k]), -1))

    # Electric vehicle simulation module
    elif car.powertrain == 'electric engine':

        curves = driver.ev_curves(
            car,
            veh_mass,
            f0,
            f1,
            f2,
            mh_base
        )

        veh_model_speed = list(np.arange(0, veh_max_speed + 0.1, 0.1))  # m/s
        veh_model_acc = []
        veh_model_dec = []
        ppar = [ppar0, ppar1, ppar2]
        dec_curves = np.poly1d(ppar)
        for k in range(len(veh_model_speed)):
            veh_model_acc.append(float(curves[0](veh_model_speed[k])))
            veh_model_dec.append(min(dec_curves(veh_model_speed[k]),-1))

    # Other vehicles simulation module
    else:
        print("The simulation module for your selected vehicle type (",
              car.powertrain,
              ") is under development!")
        exit()

    mfc_curves = {
        'mfc_speed': list(veh_model_speed),
        'mfc_acc': list(veh_model_acc),
        'mfc_dec': list(veh_model_dec),
        'mfc_f_0': f0,
        'mfc_f_1': f1,
        'mfc_f_2': f2,
        'car_length': car.car_length,
        'car_width': car.car_width,
        'car_height': car.car_height,
        'car_mass': car.veh_mass,
        'car_phi': car.phi,
        'car_wheelbase': car.wheelbase
    }

    return mfc_curves

