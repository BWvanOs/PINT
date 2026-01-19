import numpy as np

def winsor_quantiles(arr: np.ndarray, lo: float, hi: float):
    """
    Return winzorization values (q_lo, q_hi) associated with the winsorization input of the array
    So input eg 0.01, 0.99, output are associated values of the input array
    Reutrn quantiles; None if invalid input.
    Used be the Shiny app to display global range
    """
    try:
        #selected winsorization parameters
        #Uses numpy np.nanquantile which is robust to NaN input
        qlo, qhi = np.nanquantile(arr, [lo, hi])
        if np.isnan(qlo) or np.isnan(qhi):
            return None
        ##Returns the values as float
        return float(qlo), float(qhi)
    except Exception:
        return None