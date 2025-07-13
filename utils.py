# H-BAV/utils.py

def get_epsilon_exp(eps, eps_end, decay_factor):
    """Exponentialer Epsilon-Decay."""
    return max(eps * decay_factor, eps_end)

def get_epsilon_lin(eps, eps_end, decay_factor):
    """Linearer Epsilon-Decay."""
    return max(eps - decay_factor, eps_end)