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

def generate_recipe_sample():
    # Step 1: Randomly sample input features
    temp = np.random.uniform(150, 500)
    weight = np.random.uniform(50, 600)
    moisture = np.random.uniform(0.05, 0.75)
    
    umami = np.random.uniform(0, 10)
    sweet = np.random.uniform(0, 10)
    sour = np.random.uniform(0, 10)
    spice = np.random.uniform(0, 10)

    fat_g = np.random.uniform(0, 40)
    carb_g = np.random.uniform(0, 100)
    protein_g = np.random.uniform(0, 50)

    # Step 2: Use physics‑rules to compute outputs
    cook_time = cook_time_physics(temp, weight, moisture)
    flavor = flavor_intensity(umami, sweet, sour, spice)
    calories = calories_from_macros(fat_g, carb_g, protein_g)

    # Step 3: Return well‑formatted sample
    return {
        "temp": temp,
        "weight": weight,
        "moisture": moisture,
        "umami": umami,
        "sweet": sweet,
        "sour": sour,
        "spice": spice,
        "fat_g": fat_g,
        "carb_g": carb_g,
        "protein_g": protein_g,
        "cook_time": cook_time,
        "flavor": flavor,
        "calories": calories
    }
