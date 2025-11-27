import numpy as np

def cook_time_physics(temp, weight, moisture):
    return (weight * 0.8) + (moisture * 5) + (300 / (temp + 1))

def flavor_intensity(umami, sweet, sour, spice):
    return (
        0.6*umami +
        0.4*sweet -
        0.2*sour +
        1.1*spice +
        np.random.normal(0, 0.05)
    )

def calories_from_macros(fat_g, carb_g, protein_g):
    return (
        9 * fat_g +
        4 * carb_g +
        4 * protein_g +
        np.random.normal(0, 5)
    )
