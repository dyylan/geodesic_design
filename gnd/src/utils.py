import numpy as np

invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

def multikron(matrices):
    product = matrices[0]
    for mat in matrices[1:]:
        product = np.kron(product, mat)
    return product

def unitary_fidelity(unitary1, unitary2):
    return np.abs(np.trace(unitary1.conj().T @ unitary2))/len(unitary1[0])

def golden_section_search(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    
    Example:
    f = lambda x: (x-2)**2
    a = 1
    b = 5
    tol = 1e-5
    (c,d) = gss(f, a, b, tol)
    print(c, d)
    1.9999959837979107 2.0000050911830893
    source: https://en.wikipedia.org/wiki/Golden-section_search
    """

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))
    
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    
    for k in range(n-1):
        if yc > yd: 
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return c, yc
    else:
        return d, yd