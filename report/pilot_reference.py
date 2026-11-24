"""Independent sine-series solution of the exercise PDE (zero initial data)."""

import numpy as np


def pde_reference(grid_x, grid_y, times, modes=96, input_kind="sine", tf=1.):
    k = np.arange(1, modes+1)
    # Exact sine coefficients of the separable spatial source.
    bx = np.sinc(1.5-k)-np.sinc(1.5+k)
    by = np.sinc(4/3-k)-np.sinc(4/3+k)
    lam = np.pi**2*(k[:,None]**2+k[None,:]**2)
    t = np.asarray(times)[:,None,None]
    if input_kind == "sine":
        w = 2*np.pi
        response = 10*(lam*np.sin(w*t)-w*np.cos(w*t)+w*np.exp(-lam*t))/(lam**2+w**2)
    else:
        stop = 0.2*tf
        response = 10*(-np.expm1(-lam*np.minimum(t,stop)))/lam*np.exp(-lam*np.maximum(t-stop,0))
    modes_at_t = response*(by[:,None]*bx[None,:])
    sx = np.sin(np.pi*np.outer(grid_x,k))
    sy = np.sin(np.pi*np.outer(grid_y,k))
    fields = (sy[None,:,:] @ modes_at_t) @ sx.T
    return fields.transpose(1,2,0)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from visualize_system import simulate
    grid=np.linspace(0,1,19)
    times=np.linspace(0,1,101)
    a=pde_reference(grid,grid,times,96)
    b=pde_reference(grid,grid,times,192)
    fine=simulate(17)[6][:,::10].reshape(17,17,101)
    print('Reference 96 vs 192:',np.max(np.abs(a-b)))
    print('Fine FDM17 vs PDE192:',np.max(np.abs(fine-b[1:-1,1:-1])))
