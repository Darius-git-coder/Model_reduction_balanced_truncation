"""Estimator/bound-driven resolution selection; independent validation only after selection."""

import json
import numpy as np
from threadpoolctl import threadpool_limits

from experiments import (DATA,measured,save_csv,fdm_2d_poisson,balancing,euler,
                         interpolate,prepare,propagate,pde_reference)

TIMES=np.linspace(0,1,1001)
INPUTS=10*np.sin(2*np.pi*TIMES)
VALIDATION=np.linspace(0,1,19)
OBS_TIMES=TIMES[::10]
INPUT_NORM=np.sqrt(50.)
SAFETY=2.


def energy_norm(field):
    """Quadrature for L2(time; discrete spatial L2) on fixed common nodes."""
    return float(np.sqrt(np.trapz(np.sum(field**2,axis=0)/(18**2),OBS_TIMES)))


def solve_grid(intervals,steps,work):
    d=intervals-1
    A,B,*_=fdm_2d_poisson(d)
    t=np.linspace(0,1,steps+1)
    x=euler(A,B,t,10*np.sin(2*np.pi*t))
    g=np.linspace(0,1,intervals+1)
    values=interpolate(x,g,g,VALIDATION,slice(None,None,steps//100))
    work.append((d*d,steps))
    return dict(A=A,B=B,gx=g,gy=g,x=x,values=values,states=d*d)


def select_fdm(tolerance):
    work=[]
    coarse=solve_grid(3,1000,work)
    history=[]
    intervals=6
    for _ in range(5):
        fine=solve_grid(intervals,1000,work)
        spatial=energy_norm(fine['values']-coarse['values'])/3
        time_fine=solve_grid(intervals,2000,work)
        temporal=2*energy_norm(fine['values']-time_fine['values'])
        estimator=SAFETY*(spatial+temporal)
        history.append(dict(states=fine['states'],space_estimate=spatial,time_estimate=temporal,estimate=estimator))
        if estimator<=tolerance:
            return dict(**fine,estimate=estimator,history=history,work=work,success=True,
                        lyapunov_solves=0,svds=0)
        coarse=fine
        intervals*=2
    return dict(success=False,history=history,estimate=estimator)


def select_bt(tolerance):
    work=[]
    coarse=solve_grid(18,1000,work)
    fine=solve_grid(36,1000,work)
    time_fine=solve_grid(18,2000,work)
    # Coarse h-error ≈ 4/3*(coarse-fine); coarse dt-error ≈ 2*(dt-dt/2).
    spatial=4/3*energy_norm(fine['values']-coarse['values'])
    temporal=2*energy_norm(time_fine['values']-coarse['values'])
    floor=SAFETY*(spatial+temporal)
    A,B=coarse['A'],coarse['B']
    Ab,Bb,Cb,sigma=balancing(A,B,np.eye(289),1e-14)
    tails=2*np.array([np.sum(sigma[r:]) for r in range(1,len(sigma)+1)])
    # C=I was used for balancing. Multiplication by h=1/18 converts the
    # Euclidean output norm to the common physical spatial quadrature norm.
    energy_bounds=tails/18*INPUT_NORM
    eligible=np.flatnonzero(energy_bounds+floor<=tolerance)
    history=[dict(states=r+1,bt_bound=float(value),fdm_estimate=floor,
                  estimate=float(value+floor)) for r,value in enumerate(energy_bounds)]
    if not len(eligible):return dict(success=False,history=history,estimate=float(floor))
    r=int(eligible[0])+1
    xr=euler(Ab[:r,:r],Bb[:r],TIMES,INPUTS)
    work.append((r,1000))
    values=np.pad((Cb[:,:r]@xr[:,::10]).reshape(17,17,101),((1,1),(1,1),(0,0))).reshape(361,101)
    return dict(A=Ab[:r,:r],B=Bb[:r],C=Cb[:,:r],x=xr,values=values,states=r,
                estimate=float(energy_bounds[r-1]+floor),bt_bound=float(energy_bounds[r-1]),
                fdm_estimate=float(floor),history=history,work=work,success=True,
                lyapunov_solves=2,svds=1)


def main():
    # The reference is deliberately absent from select_fdm/select_bt arguments.
    reference=pde_reference(VALIDATION,VALIDATION,OBS_TIMES,192).reshape(361,101)
    rows=[]
    failures=[]
    for tol in [.01,.005,.002,.001,.0005]:
        for method,selector in [('FDM',select_fdm),('BT',select_bt)]:
            result,timing=measured(lambda:selector(tol),repeats=3)
            if not result['success']:
                print(method,tol,'not attainable',result['estimate'],flush=True)
                failures.append(dict(method=method,tolerance=tol,estimated_floor=result['estimate'],
                    adaptation=timing['median'],reason='Fixed fine FDM model exhausts the error budget; refine h and rebalance'))
                continue
            error=energy_norm(result['values']-reference)
            maximum=float(np.max(np.abs(result['values']-reference)))
            step=prepare(result['A'],.001)
            state,on=measured(lambda:propagate(step,result['B'],TIMES,INPUTS))
            if method=='BT':
                _,output=measured(lambda:result['C']@state[:,::10])
            else:
                _,output=measured(lambda:interpolate(state,result['gx'],result['gy'],VALIDATION,slice(None,None,10)))
            row=dict(method=method,tolerance=tol,states=result['states'],estimate=result['estimate'],
                error=error,max_error=maximum,effectivity=result['estimate']/error,
                adaptation=timing['median'],adapt_q25=timing['q25'],adapt_q75=timing['q75'],
                online=on['median'],output=output['median'],
                trial_simulations=len(result['work']),linear_solves=sum(s for n,s in result['work']),
                factorizations=len(result['work']),scalar_state_updates=sum(n*s for n,s in result['work']),
                lyapunov_solves=result['lyapunov_solves'],svds=result['svds'])
            rows.append(row)
            print(row,flush=True)
            (DATA/f'bounds_{method}_{tol:g}.json').write_text(json.dumps(result['history'],indent=2))
            if tol==.002:
                np.savez_compressed(DATA/f'bounds_{method}.npz',values=result['values'],
                    validation=VALIDATION,reference=reference,times=OBS_TIMES,
                    gx=result.get('gx',VALIDATION),gy=result.get('gy',VALIDATION))
    save_csv('bounds_adaptive.csv',rows)
    if failures:save_csv('bounds_failures.csv',failures)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
