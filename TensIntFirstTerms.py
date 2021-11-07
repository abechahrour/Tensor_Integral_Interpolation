
import numpy as np
Pi = np.float64(3.14159265358979323846)



def Log(a):
    return np.log(a)

def k1FirstTerm(t, m):
    t = t[:, np.newaxis]
    m = m[:, np.newaxis]
    p1 = np.array([0.5, 0, 0.5], dtype = np.float64)
    p2 = np.array([0.5, 0, -0.5], dtype = np.float64)
    ct = 2*t+1
    st = np.sqrt(1 - ct**2)
    p3 = -np.array([[0.5, 0.5*np.sqrt(1 - ct**2), 0.5*ct] for ct in ct], dtype = np.float64)
    p4 = -np.array([[0.5, -0.5*np.sqrt(1 - ct**2), -0.5*ct]for ct in ct], dtype = np.float64)
    p1 = np.tile(p1, (t.shape[0], 1))
    p2 = np.tile(p2, (m.shape[0], 1))

    return ((p2*(Pi**2 - 2*Log(m**2)**2 + Log(m**2)*Log(t**2)))/(2.*t) +
            (p4*(4*Pi**2*(1 + t) - 4*(1 + t)*Log(m**2)**2 + Log(t**2)**2))/(8.*t*(1 + t)) +
            (p1*(8*Pi**2*(1 + t) - 12*(1 + t)*Log(m**2)**2 +
            4*(1 + t)*Log(m**2)*Log(t**2) + Log(t**2)**2))/(8.*t*(1 + t)))
