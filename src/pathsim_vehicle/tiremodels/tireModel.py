#########################################################################################
##
##                  Tire model base contract
##
#########################################################################################



# TIRE MODEL Definitions ================================================================

class TireModel:
    """Base contract of the tire-model ladder (a plain class, not a block).

    Concrete models implement the single stateless method
    ``forces(kappa, alpha, F_z) -> (F_x_t, F_y_t)``.
    """

    def forces(self, kappa, alpha, F_z):
        """Tire-frame contact forces at the given slip state and load.

        Parameters
        ----------
        kappa : float
            Longitudinal slip ratio [-].
        alpha : float
            Slip angle [rad].
        F_z : float
            Vertical load [N], clamped non-negative by the caller.

        Returns
        -------
        F_x_t : float
            Tire-frame longitudinal force [N].
        F_y_t : float
            Tire-frame lateral force [N].
        """
        raise NotImplementedError
