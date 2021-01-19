import os
import car_following_env.vehicle_dynamics.mfc_model.reading_n_organizing as rno


def get_veh_parm(car_id, veh_load):
    db_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'car_database', '2019_07_03_car_db')
    db = rno.load_db_to_dictionary(db_name)
    car = rno.get_vehicle_from_db(db, car_id)
    veh_mass = car.veh_mass + veh_load
    # phi = 1.03
    return [car.car_width, car.car_height, veh_mass, car.phi]

