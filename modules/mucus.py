# modules/mucus.py

class MucusDynamics:
    """
    Simple mucus dynamics model.
    - Adds an extra resistance term R_m(t) depending on mucus burden M(t).
    - M(t) evolves over time, balancing mucus production and clearance by airflow.
    """

    def __init__(self, M0=0.2, Rm0=0.003, alpha=1.0, k_prod=0.001, k_clear=0.05):
        self.M0 = float(M0)
        self.M = float(M0)
        self.Rm0 = float(Rm0)
        self.alpha = float(alpha)
        self.k_prod = float(k_prod)
        self.k_clear = float(k_clear)

    def reset(self):
        self.M = float(self.M0)

    def update(self, Q, dt):
        shear = abs(Q) / (1.0 + abs(Q))  # shear ranges 0..1
        dM = self.k_prod - self.k_clear * shear * self.M
        self.M = max(0.0, min(1.0, self.M + dM * dt))

    def resistance(self):
        return self.Rm0 * (1.0 + self.alpha * self.M)


