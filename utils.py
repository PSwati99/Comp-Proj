def transmission_coefficient(f_GHz):
    """Returns the transmission coefficient based on frequency."""
    return 0.5 if f_GHz < 55 else 1.0

def spin_wave_amplitude(f_GHz):
    """
    Simple model for spin-wave amplitude ρ that decreases with frequency.
    """
    return 1.0 / (1 + f_GHz / 50)
