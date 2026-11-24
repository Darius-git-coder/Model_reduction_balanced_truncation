"""Implicit Euler with measured accept/reject counts and explicit work models."""

from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse.linalg import splu


class WorkSolver:
    """Count solves/factorizations; approximate structural arithmetic, not CPU FLOPs."""
    def __init__(self,A):
        self.A=A
        self.n=A.shape[0]
        self.cache=OrderedDict()
        self.factors=0
        self.solves=0
        self.solve_flops=0.
        self.factor_flops=0.

    def solver(self,dt):
        key=float(dt).hex()
        if key in self.cache:
            return self.cache[key]
        n=self.n
        if sp.issparse(self.A):
            lu=splu((sp.eye(n,format='csc')-dt*self.A).tocsc())
            step=lu.solve
            arithmetic=2*(lu.L.nnz+lu.U.nnz)-3*n
            lower=np.diff(lu.L.tocsc().indptr)-1
            upper=np.diff(lu.U.tocsr().indptr)-1
            self.factor_flops+=float(np.sum(2*lower*upper+lower))
        else:
            lu=lu_factor(np.eye(n)-dt*self.A)
            step=lambda rhs:lu_solve(lu,rhs,check_finite=False)
            arithmetic=2*n*n-n
            self.factor_flops+=2/3*n**3
        self.factors+=1
        self.cache[key]=(step,arithmetic)
        if len(self.cache)>16:self.cache.popitem(last=False)
        return self.cache[key]

    def solve(self,dt,rhs):
        step,arithmetic=self.solver(dt)
        self.solves+=1
        self.solve_flops+=arithmetic
        return step(rhs)


def adaptive_euler(A,B,C,rtol=1e-3,atol=1e-5,tf=1.):
    """Common physical-output weighted RMS error control for FDM and BT.

    One full step and two half steps estimate the local error. The two-half-
    step result is accepted without Richardson extrapolation (still order 1).
    A step may be rejected. Accepted midpoints are stored for interpolation.
    """
    solver=WorkSolver(A)
    x=np.zeros(A.shape[0])
    t,h=0.,.02
    times=[t]
    states=[x.copy()]
    accepted=rejected=0
    attempts=[]
    rhs_builds=input_evals=0
    output_flops=0
    b=B[:,0]
    observe=(lambda v:v) if C is None else (lambda v:C@v)
    while t<tf-1e-14:
        h=min(h,tf-t)
        u_end=10*np.sin(2*np.pi*(t+h))
        u_mid=10*np.sin(2*np.pi*(t+h/2))
        full=solver.solve(h,x+h*b*u_end)
        half=solver.solve(h/2,x+h/2*b*u_mid)
        fine=solver.solve(h/2,half+h/2*b*u_end)
        rhs_builds+=3
        input_evals+=2
        old_output,full_output,fine_output=observe(x),observe(full),observe(fine)
        if C is not None:output_flops+=3*C.shape[0]*(2*C.shape[1]-1)
        scale=atol+rtol*np.maximum(np.abs(old_output),np.abs(fine_output))
        eta=np.sqrt(np.mean(((fine_output-full_output)/scale)**2))
        ok=bool(eta<=1)
        attempts.append((t,h,eta,ok))
        if ok:
            times.extend([t+h/2,t+h])
            states.extend([half.copy(),fine.copy()])
            t+=h
            x=fine
            accepted+=1
        else:
            rejected+=1
        factor=3. if eta==0 else float(np.clip(.9*eta**(-.5),.2,3.))
        h*=factor if ok else min(.9,factor)
        if len(attempts)>20000 or h<1e-13:raise RuntimeError('Adaptive time loop did not converge')
    return dict(times=np.array(times),states=np.array(states).T,attempts=np.array(attempts),
        accepted=accepted,rejected=rejected,solves=solver.solves,factorizations=solver.factors,
        input_evals=input_evals,rhs_builds=rhs_builds,solve_flops=solver.solve_flops,
        factor_flops=solver.factor_flops,output_flops=output_flops,
        rhs_flops=rhs_builds*(3*A.shape[0]))


def fixed_euler(A,B,C,steps=1000,tf=1.):
    solver=WorkSolver(A)
    times=np.linspace(0,tf,steps+1)
    x=np.zeros(A.shape[0])
    states=np.zeros((len(x),len(times)))
    h=tf/steps
    for k in range(steps):
        x=solver.solve(h,x+h*B[:,0]*10*np.sin(2*np.pi*times[k+1]))
        states[:,k+1]=x
    return dict(times=times,states=states,accepted=steps,rejected=0,
        solves=solver.solves,factorizations=solver.factors,input_evals=steps,rhs_builds=steps,
        solve_flops=solver.solve_flops,factor_flops=solver.factor_flops,output_flops=0,
        rhs_flops=steps*3*A.shape[0])
