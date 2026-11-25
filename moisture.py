"""One-way sealed-box moisture balance driven by thermal FEM output."""

import numpy as np
from scipy.integrate import solve_ivp


VAPOR_R = 461.526
TRANSFER_M_S = 1e-3
PIZZA_INITIAL_KG = 0.750
PIZZA_WATER_KG = PIZZA_INITIAL_KG * 0.40
PIZZA_DRY_KG = PIZZA_INITIAL_KG - PIZZA_WATER_KG
PIZZA_EXPOSED_M2 = np.pi * (0.15**2 + 0.30 * 0.015)


def saturation_pa(temperature_c):
    return 610.78 * np.exp(17.2694 * temperature_c / (temperature_c + 237.29))


def water_fraction_percent(retention_percent):
    """Convert relative water retention to the wet-basis pizza mass percentage."""
    water_kg = PIZZA_WATER_KG * retention_percent / 100
    return 100 * water_kg / (PIZZA_DRY_KG + water_kg)


def moisture_retention(thermal, air_volume_m3):
    """Use Tetens, mean film temperature, and evaporation/condensation-only fluxes."""
    times = thermal.wall_sample_s
    air = np.interp(times, thermal.time_s, thermal.air_c)
    pizza = np.interp(times, thermal.time_s, thermal.pizza_c)
    wall = thermal.wall_face_c.astype(np.float64)
    pressure = saturation_pa(wall)
    order = np.argsort(pressure, axis=1)
    pressure = np.take_along_axis(pressure, order, axis=1)
    weights = TRANSFER_M_S * thermal.wall_face_area_m2 / (VAPOR_R * (273.15 + (wall + air[:, None]) / 2))
    weights = np.take_along_axis(weights, order, axis=1)
    # Sum positive wall fluxes using cumulative slopes and offsets at each time.
    slopes = np.pad(np.cumsum(weights, axis=1), ((0, 0), (1, 0)))
    offsets = np.pad(np.cumsum(weights * pressure, axis=1), ((0, 0), (1, 0)))

    def wall_flux(index, vapor_pressure):
        count = np.searchsorted(pressure[index], vapor_pressure, side="right")
        return max(0.0, vapor_pressure * slopes[index, count] - offsets[index, count])

    def balance(time, water):
        index = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
        mix = (time - times[index]) / (times[index + 1] - times[index])
        air_c = air[index] + mix * (air[index + 1] - air[index])
        pizza_c = pizza[index] + mix * (pizza[index + 1] - pizza[index])
        vapor_pressure = max(0.0, water[1]) * VAPOR_R * (air_c + 273.15) / air_volume_m3
        film_k = 273.15 + (pizza_c + air_c) / 2
        evaporation = TRANSFER_M_S * PIZZA_EXPOSED_M2 / (VAPOR_R * film_k) * max(
            0.0, saturation_pa(pizza_c) - vapor_pressure
        )
        condensation = (1 - mix) * wall_flux(index, vapor_pressure) + mix * wall_flux(index + 1, vapor_pressure)
        return [-evaporation, evaporation - condensation, condensation]

    vapor_initial = 0.5 * saturation_pa(22) * air_volume_m3 / (VAPOR_R * 295.15)
    initial = [PIZZA_WATER_KG, vapor_initial, 0.0]
    result = solve_ivp(balance, (times[0], times[-1]), initial, method="Radau",
                       rtol=1e-6, atol=1e-10, max_step=4.0)
    if not result.success:
        raise ValueError(result.message)
    if result.y.min() < -1e-9 or np.max(np.abs(result.y.sum(axis=0) - sum(initial))) > 1e-8:
        raise ValueError("Moisture balance failed positivity or mass conservation.")
    return float(100 * result.y[0, -1] / PIZZA_WATER_KG)
