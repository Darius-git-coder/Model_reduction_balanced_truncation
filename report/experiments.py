"""Reproducible experiments for the German FDM/BT report.

Run from the repository: OPENBLAS_NUM_THREADS=1 python report/experiments.py
All times are wall-clock seconds, warm-up + medians, no plotting inside timings.
"""

import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.fft import dstn, idstn
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers import balancing, fdm_2d_poisson
from pilot_reference import pde_reference

HERE = Path(__file__).resolve().parent
DATA = HERE/"data"
DATA.mkdir(exist_ok=True)


def measured(fun, repeats=5):
    fun()
    samples=[]
    result=None
    for _ in range(repeats):
        start=perf_counter()
        result=fun()
        samples.append(perf_counter()-start)
    return result, dict(median=float(np.median(samples)), q25=float(np.quantile(samples,.25)),
                        q75=float(np.quantile(samples,.75)), samples=samples)


def euler(A, B, times, inputs):
    """Sparse LU for FDM; dense LU for small dense reduced matrices."""
    dt=times[1]-times[0]
    if sp.issparse(A):
        solver=splu((sp.eye(A.shape[0],format="csc")-dt*A).tocsc())
        step=solver.solve
    else:
        factors=lu_factor(np.eye(A.shape[0])-dt*A)
        step=lambda rhs:lu_solve(factors,rhs,check_finite=False)
    return propagate(step,B,times,inputs)


def prepare(A,dt):
    if sp.issparse(A):
        return splu((sp.eye(A.shape[0],format="csc")-dt*A).tocsc()).solve
    factors=lu_factor(np.eye(A.shape[0])-dt*A)
    return lambda rhs:lu_solve(factors,rhs,check_finite=False)


def propagate(step,B,times,inputs):
    x=np.zeros((B.shape[0],len(times)))
    b=np.asarray(B).ravel()
    dt=times[1]-times[0]
    for j in range(1,len(times)):
        x[:,j]=step(x[:,j-1]+dt*b*inputs[j])
    return x


def exact_discrete(d,times):
    A,B,C,X,Y=fdm_2d_poisson(d)
    k=np.arange(1,d+1)
    lam=4*(d+1)**2*(np.sin(np.pi*k[:,None]/(2*(d+1)))**2+np.sin(np.pi*k[None,:]/(2*(d+1)))**2)
    t=np.asarray(times)[None,None,:]
    lam=lam[:,:,None]
    w=2*np.pi
    response=10*(lam*np.sin(w*t)-w*np.cos(w*t)+w*np.exp(-lam*t))/(lam**2+w**2)
    bhat=dstn(B.reshape(d,d),type=1,norm="ortho")
    return idstn(bhat[:,:,None]*response,type=1,norm="ortho",axes=(0,1)).reshape(d*d,len(times))


def nonuniform_system(gx,gy):
    def second(grid):
        left=np.diff(grid)[:-1]
        right=np.diff(grid)[1:]
        diag=-2/(left*right)
        low=2/(left*(left+right))
        high=2/(right*(left+right))
        return sp.diags([low[1:],diag,high[:-1]],[-1,0,1],format="csc")
    nx,ny=len(gx)-2,len(gy)-2
    A=sp.kron(sp.eye(ny),second(gx))+sp.kron(second(gy),sp.eye(nx))
    X,Y=np.meshgrid(gx[1:-1],gy[1:-1])
    B=(np.sin(1.5*np.pi*X)*np.sin(4/3*np.pi*Y)).reshape(-1,1)
    return A.tocsc(),B


def interpolate(x,gx,gy,validation,sample=slice(None)):
    field=x[:,sample].reshape(len(gy)-2,len(gx)-2,-1)
    field=np.pad(field,((1,1),(1,1),(0,0)))
    X,Y=np.meshgrid(validation,validation)
    points=np.column_stack([Y.ravel(),X.ravel()])
    return RegularGridInterpolator((gy,gx),field)(points)


def save_csv(name,rows):
    with (DATA/name).open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    with threadpool_limits(limits=1):
        times=np.linspace(0,1,1001)
        inputs=10*np.sin(2*np.pi*times)
        validation=np.linspace(0,1,19)
        reference=pde_reference(validation,validation,times[::10],96)
        refined_reference=pde_reference(validation,validation,times[::10],192)
        ref_delta=float(np.max(np.abs(reference-refined_reference)))
        reference=refined_reference.reshape(19*19,101)
        env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                 platform=platform.platform(),machine=platform.machine(),threads=threadpool_info(),
                 date="2026-11-24",reference_modes=192,reference_mode_check=ref_delta,
                 validation_grid=19,validation_times=101,timing_repeats=5)
        (DATA/"environment.json").write_text(json.dumps(env,indent=2))
        print('Reference check',ref_delta,flush=True)

        print('Runtime scaling',flush=True)
        timing_rows=[]
        cache={}
        for d in [5,8,12,17,24]:
            system,asm=measured(lambda:fdm_2d_poisson(d))
            A,B,C,*_=system
            balanced,offline=measured(lambda:balancing(A,B,np.eye(d*d),1e-14),repeats=3)
            Ab,Bb,Cb,sigma=balanced
            cache[d]=balanced
            fstep,ffact=measured(lambda:prepare(A,.001))
            bstep,bfact=measured(lambda:prepare(Ab[:4,:4],.001))
            xf,ftime=measured(lambda:propagate(fstep,B,times,inputs))
            xb,btime=measured(lambda:propagate(bstep,Bb[:4],times,inputs))
            _,output=measured(lambda:Cb[:,:4]@xb)
            row=dict(d=d,n=d*d,nnz=A.nnz,assembly=asm["median"],bt_build=offline["median"],
                fdm_factor=ffact["median"],bt_factor=bfact["median"],
                fdm_online=ftime["median"],bt_online=btime["median"],bt_reconstruct=output["median"],
                fdm_q25=ftime["q25"],fdm_q75=ftime["q75"],bt_q25=btime["q25"],bt_q75=btime["q75"],
                bt_build_q25=offline["q25"],bt_build_q75=offline["q75"],numerical_order=len(sigma))
            timing_rows.append(row)
            print(row,flush=True)
        save_csv("runtime.csv",timing_rows)

        print('Space and time convergence',flush=True)
        # An exact discrete sine mode separates spatial and temporal errors.
        spatial=[]
        for d in [3,7,15,31,63]:
            h=1/(d+1)
            X,Y=np.meshgrid(np.arange(1,d+1)*h,np.arange(1,d+1)*h)
            mode=np.sin(np.pi*X)*np.sin(np.pi*Y)
            lam_h=8/h**2*np.sin(np.pi*h/2)**2
            t=.1
            error=np.max(np.abs(mode*(np.exp(-lam_h*t)-np.exp(-2*np.pi**2*t))))
            spatial.append(dict(d=d,h=h,error=error))
        save_csv("space_convergence.csv",spatial)
        temporal=[]
        d=17
        exact=exact_discrete(d,[1.])[:,0]
        A,B,*_=fdm_2d_poisson(d)
        for steps in [50,100,200,400,800,1600]:
            ts=np.linspace(0,1,steps+1)
            x=euler(A,B,ts,10*np.sin(2*np.pi*ts))
            temporal.append(dict(steps=steps,dt=1/steps,error=float(np.max(np.abs(x[:,-1]-exact)))))
        save_csv("time_convergence.csv",temporal)

        print('Rank, frequency bounds, output dependence',flush=True)
        Ab,Bb,Cb,sigma=cache[17]
        A,B,C1,*_=fdm_2d_poisson(17)
        lam,V=np.linalg.eigh(A.toarray())
        modal_b=V.T@B
        omega=np.r_[0.,np.logspace(-5,5,1000)]
        full=V@(modal_b/(1j*omega[None,:]-lam[:,None]))
        fine=euler(A,B,times,inputs)
        reference_state=reference.reshape(19,19,101)[1:-1,1:-1].reshape(289,101)
        ranks=[]
        freq_curves={}
        for r in range(1,min(12,len(sigma)-1)+1):
            Ar,Br,Cr=Ab[:r,:r],Bb[:r],Cb[:,:r]
            xr=euler(Ar,Br,times,inputs)
            field=Cr@xr
            reduced=np.column_stack([Cr@np.linalg.solve(1j*w*np.eye(r)-Ar,Br)[:,0] for w in omega])
            response=np.linalg.norm(full-reduced,axis=0)
            ranks.append(dict(r=r,error_fdm_reference=float(np.max(np.abs(field-fine))),
                error_pde_reference=float(np.max(np.abs(field[:,::10]-reference_state))),
                sampled_hinf=float(np.max(response)),tail_bound=float(2*np.sum(sigma[r:])),
                stability=float(np.max(np.linalg.eigvals(Ar).real))))
            if r in [2,4,8]:freq_curves[str(r)]=response
        save_csv("rank_study.csv",ranks)
        Ab1,Bb1,Cb1,sigma1=balancing(A,B,C1,1e-14)
        np.savez_compressed(DATA/"spectra_frequency.npz",sigma=sigma,sigma_avg=sigma1,omega=omega,**freq_curves)
        pde_floor=float(np.max(np.abs(fine[:,::10]-reference_state)))
        (DATA/"reference_summary.json").write_text(json.dumps(dict(pde_floor=pde_floor,reference_check=ref_delta),indent=2))

        from adaptive_experiments import run_adaptive_studies
        run_adaptive_studies()
        print('All datasets completed',flush=True)


if __name__=='__main__':
    main()
