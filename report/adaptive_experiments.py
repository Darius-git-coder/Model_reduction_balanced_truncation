"""Fair fixed/adaptive studies with timing, work counters and accept/reject logs."""


import numpy as np
from scipy.interpolate import interp1d
from threadpoolctl import threadpool_limits

from experiments import DATA,measured,save_csv,fdm_2d_poisson,balancing,exact_discrete
from adaptive_time import adaptive_euler,fixed_euler


def run_time_study():
    times=np.linspace(0,1,1001)
    print('Fixed/adaptive time: same output-weighted Euler controller',flush=True)
    A,B,*_=fdm_2d_poisson(17)
    Ab,Bb,Cb,sigma=balancing(A,B,np.eye(289),1e-14)
    time_rows=[]
    for method,mat,bmat,C in [('FDM',A,B,None),('BT',Ab[:4,:4],Bb[:4],Cb[:,:4])]:
        physical=(lambda values:values) if C is None else (lambda values:C@values)
        if method=='FDM':
            exact=exact_discrete(17,times[::10])
        else:
            vals,vecs=np.linalg.eig(mat)
            modal=np.linalg.solve(vecs,bmat)[:,0]
            t=times[::10][None,:]
            lam=-vals[:,None]
            response=10*(lam*np.sin(2*np.pi*t)-2*np.pi*np.cos(2*np.pi*t)+2*np.pi*np.exp(-lam*t))/(lam**2+(2*np.pi)**2)
            exact=(C@vecs@(modal[:,None]*response)).real
        cases=[(f'fixed_{steps}',lambda steps=steps:fixed_euler(mat,bmat,C,steps),steps,0.) for steps in [100,300,1000]]
        cases += [(f'adaptive_{tol:g}',lambda tol=tol:adaptive_euler(mat,bmat,C,rtol=tol,atol=tol*.01),0,tol)
                  for tol in [1e-2,1e-3,1e-4]]
        for name,fun,fixed_steps,tol in cases:
            result,timing=measured(fun,repeats=5)
            values=interp1d(result['times'],result['states'],axis=1)(times[::10])
            error=float(np.max(np.abs(physical(values)-exact)))
            row=dict(method=method,mode='fixed' if fixed_steps else 'adaptive',setting=name,
                rtol=tol,steps=fixed_steps,seconds=timing['median'],q25=timing['q25'],q75=timing['q75'],
                error_time_only=error,accepted=result['accepted'],rejected=result['rejected'],
                solves=result['solves'],factorizations=result['factorizations'],input_evals=result['input_evals'],
                rhs_builds=result['rhs_builds'],solve_flops=result['solve_flops'],factor_flops=result['factor_flops'],
                output_flops=result['output_flops'],rhs_flops=result['rhs_flops'])
            time_rows.append(row)
            print(row,flush=True)
            if not fixed_steps and tol==1e-3:
                np.savez_compressed(DATA/f'adaptive_time_{method}.npz',times=result['times'],attempts=result['attempts'])
    save_csv('adaptive_time.csv',time_rows)

def run_adaptive_studies():
    from bound_adaptivity import main
    main()
    run_time_study()



if __name__=='__main__':
    with threadpool_limits(limits=1):
        run_adaptive_studies()
