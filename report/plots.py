"""Publication-style vector figures and tables from measured CSV/NPZ data."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE))
from compare_methods import compare,snapshot_fields

DATA=HERE/'data'
FIG=HERE/'figures'
FIG.mkdir(exist_ok=True)
BLUE='#2864a3'; PINK='#b02b80'; GREY='#62717b'; ORANGE='#d87922'
COLORS={'FDM':BLUE,'BT':PINK}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.titlesize':13.5,
    'axes.labelsize':12.5,'legend.fontsize':11,'figure.dpi':130,'savefig.dpi':180,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
    'grid.alpha':.18,'grid.linewidth':.6,'axes.axisbelow':True,'pdf.fonttype':42})


def read(name):
    with (DATA/name).open() as file:
        rows=list(csv.DictReader(file))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except ValueError:pass
    return rows


def column(rows,key):return np.array([r[key] for r in rows])


def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    plt.close(fig)


def main():
    spatial=read('space_convergence.csv'); temporal=read('time_convergence.csv')
    timing=read('runtime.csv'); ranks=read('rank_study.csv')
    adaptive=read('bounds_adaptive.csv'); time_adapt=read('adaptive_time.csv')
    spectra=np.load(DATA/'spectra_frequency.npz')
    ref_summary=json.loads((DATA/'reference_summary.json').read_text())

    result=compare()
    k=int(np.argmax(np.max(np.abs(result['fine'][6]),axis=0)))
    time=result['fine'][5][k]
    fields=snapshot_fields(result,k)
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.2),layout='constrained')
    titles=['Feine FDM\n289 Zustände','Grobe FDM\n64 Zustände','Balanced Truncation\n4 Zustände',
            'FDM-Abweichung\nzur feinen FDM','BT-Abweichung\nzur feinen FDM',r'$|e_{\rm FDM}|-|e_{\rm BT}|$']
    zmax=max(np.max(np.abs(v)) for v in fields[:3]); emax=max(np.max(v) for v in fields[3:5])
    for j,ax in enumerate(axes.flat):
        d=17 if j in (0,2) else 8
        grid=np.linspace(0,1,d+2)
        low,high=(-zmax,zmax) if j<3 else ((0,emax) if j<5 else (-emax,emax))
        cmap='RdBu_r' if j<3 else ('magma' if j<5 else 'PiYG')
        im=ax.pcolormesh(grid,grid,fields[j],shading='nearest',cmap=cmap,vmin=low,vmax=high)
        ax.set(title=titles[j],xlabel=r'$x_1$',ylabel=r'$x_2$',aspect='equal',xlim=(0,1),ylim=(0,1))
        ax.grid(False); fig.colorbar(im,ax=ax,shrink=.75)
    fig.suptitle(f'Lösungen und gemeinsame Errormaps bei t={time:.3f}',fontsize=12)
    save(fig,'01_fields_errors')

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    for key,label,color in [('sigma','Gesamtes Feld $C=I$',PINK),('sigma_avg','Räumliches Integral $C=C_1$',BLUE)]:
        s=spectra[key]
        axes[0].semilogy(np.arange(1,len(s)+1),s/s[0],'o-',ms=3,color=color,label=label)
    axes[0].set(xlabel='Index k',ylabel=r'$\sigma_k/\sigma_1$',title='Welche Richtungen sind wichtig?')
    axes[0].legend()
    for r,color in [(2,BLUE),(4,PINK),(8,ORANGE)]:
        w=spectra['omega'][1:]; error=spectra[str(r)][1:]
        axes[1].loglog(w,error,color=color,label=f'r={r}')
        bound=next(row['tail_bound'] for row in ranks if row['r']==r)
        axes[1].axhline(bound,color=color,ls='--',lw=.8)
    axes[1].set(xlabel=r'Kreisfrequenz $\omega$',ylabel=r'$\|G(i\omega)-G_r(i\omega)\|_2$',title='Frequenzfehler; gestrichelt: BT-Schranke')
    axes[1].legend(); save(fig,'02_hsv_frequency')

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    h=column(spatial,'h'); e=column(spatial,'error')
    axes[0].loglog(h,e,'o-',color=BLUE,label='Gemessener Raumfehler')
    axes[0].loglog(h,e[-1]*(h/h[-1])**2,'--',color=GREY,label=r'Referenz: $h^2$')
    axes[0].set(xlabel='Gitterabstand h',ylabel='Maximaler Fehler',title='Raum: zweiter Ordnung'); axes[0].legend()
    dt=column(temporal,'dt'); e=column(temporal,'error')
    axes[1].loglog(dt,e,'o-',color=PINK,label='Impliziter Euler')
    axes[1].loglog(dt,e[-1]*dt/dt[-1],'--',color=GREY,label=r'Referenz: $\Delta t$')
    axes[1].set(xlabel=r'Zeitschritt $\Delta t$',ylabel='Maximaler Endzeitfehler',title='Zeit: erster Ordnung');axes[1].legend()
    save(fig,'03_convergence')

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    r=column(ranks,'r')
    axes[0].semilogy(r,column(ranks,'error_fdm_reference'),'o-',color=PINK,label='Fehler zur feinen FDM')
    axes[0].semilogy(r,column(ranks,'error_pde_reference'),'s-',color=BLUE,label='Fehler zur Sinusreihenreferenz')
    axes[0].axhline(ref_summary['pde_floor'],ls='--',color=GREY,label='Fehler des feinen FDM-Modells')
    axes[0].set(xlabel='BT-Modellordnung r',ylabel='Maximaler Feldfehler',title='Mehr BT-Zustände beseitigen keinen Gitterfehler');axes[0].legend()
    axes[1].semilogy(r,column(ranks,'sampled_hinf'),'o-',color=PINK,label='Abgetasteter Frequenzfehler')
    axes[1].semilogy(r,column(ranks,'tail_bound'),'s--',color=BLUE,label=r'$2\sum_{k>r}\sigma_k$')
    axes[1].set(xlabel='BT-Modellordnung r',ylabel=r'$\mathcal{H}_\infty$-Fehler / Schranke',title='Normgerechte Fehlerkontrolle');axes[1].legend()
    save(fig,'04_rank_and_bounds')

    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    n=column(timing,'n'); build=column(timing,'bt_build')
    axes[0].loglog(n,build,'o-',color=PINK,label='BT-Konstruktion (dicht)')
    axes[0].loglog(n,column(timing,'assembly'),'o-',color=BLUE,label='FDM-Matrixaufbau')
    axes[0].loglog(n,build[-1]*(n/n[-1])**3,'--',color=GREY,label=r'Orientierung: $n^3$')
    axes[0].set(xlabel='Dimension des Ausgangsmodells n',ylabel='Medianzeit [s]',title='Einmalige Aufbaukosten');axes[0].legend()
    for key,color,label in [('fdm_online',BLUE,'FDM: Zustandssimulation'),('bt_online',PINK,'BT r=4: Zustandssimulation'),('bt_reconstruct',ORANGE,'BT: Feldrekonstruktion')]:
        axes[1].loglog(n,column(timing,key),'o-',color=color,label=label)
    axes[1].set(xlabel='Dimension des Ausgangsmodells n',ylabel='Medianzeit [s]',title='1 000 Zeitschritte; Faktorisierung vorab');axes[1].legend()
    save(fig,'05_runtime_scaling')

    base=next(row for row in timing if row['n']==289)
    m=np.arange(1,101)
    fdm=base['assembly']+base['fdm_factor']+m*base['fdm_online']
    bt=base['assembly']+base['bt_build']+base['bt_factor']+m*(base['bt_online']+base['bt_reconstruct'])
    breakeven=(base['bt_build']+base['bt_factor']-base['fdm_factor'])/(base['fdm_online']-base['bt_online']-base['bt_reconstruct'])
    fig,ax=plt.subplots(figsize=(8.4,3.1),layout='constrained')
    ax.plot(m,fdm,color=BLUE,label='Feine FDM n=289')
    ax.plot(m,bt,color=PINK,label='BT r=4 einschließlich Feldrekonstruktion')
    ax.axvline(breakeven,color=GREY,ls='--',label=f'Rechnerischer Schnittpunkt: {breakeven:.1f} Läufe')
    ax.set(xlabel='Anzahl der Simulationen mit unveränderten A, B, C',ylabel='Kumulierte Zeit [s]',title='Wann amortisiert sich die Balancierung?');ax.legend()
    save(fig,'06_amortization')

    fig,axes=plt.subplots(2,2,figsize=(10.2,6.4),layout='constrained')
    for method in ['FDM','BT']:
        rows=[row for row in adaptive if row['method']==method]
        color=COLORS[method]
        for ax,key,title in [(axes[0,0],'adaptation','Gesamte Auswahlphase'),(axes[0,1],'online','Neue Simulation nach Auswahl'),
                             (axes[1,0],'scalar_state_updates','Dimensionsgewichtete Lösearbeit')]:
            ax.loglog(column(rows,key),column(rows,'error'),'o-',color=color,label=method)
            ax.set(ylabel='Validierter Energienormfehler',title=title)
        axes[1,1].loglog(column(rows,'tolerance'),column(rows,'estimate'),'o--',color=color,label=f'{method}: Schätzer/Summe')
        axes[1,1].loglog(column(rows,'tolerance'),column(rows,'error'),'s-',color=color,label=f'{method}: Validierung')
    axes[0,0].set_xlabel('Medianzeit [s]');axes[0,1].set_xlabel('Medianzeit [s]')
    axes[1,0].set_xlabel(r'$\sum$ (Dimension × Zeitschritte)')
    taus=sorted({row['tolerance'] for row in adaptive})
    axes[1,1].loglog(taus,taus,':',color=GREY,label='Zieltoleranz')
    axes[1,1].set(xlabel='Zieltoleranz',ylabel='Fehler / Schätzer',title='Schranke, Schätzer und tatsächlicher Fehler')
    for ax in axes.flat:ax.legend()
    save(fig,'07_adaptive_resolution')

    f=np.load(DATA/'bounds_FDM.npz'); b=np.load(DATA/'bounds_BT.npz')
    k=int(np.argmax(np.max(np.abs(f['reference']),axis=0)))
    ef=np.abs(f['values'][:,k]-f['reference'][:,k]).reshape(19,19)
    eb=np.abs(b['values'][:,k]-b['reference'][:,k]).reshape(19,19)
    high=max(np.max(ef),np.max(eb)); grid=f['validation']
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.3),layout='constrained')
    for ax,values,title,cmap,low,up in zip(axes,[ef,eb,ef-eb],['Adaptive FDM','Adaptive BT','Positiv: BT lokal genauer'],
            ['magma','magma','PiYG'],[0,0,-high],[high,high,high]):
        im=ax.pcolormesh(grid,grid,values,shading='nearest',cmap=cmap,vmin=low,vmax=up)
        ax.set(title=title,xlabel=r'$x_1$',ylabel=r'$x_2$',aspect='equal',xlim=(0,1),ylim=(0,1));ax.grid(False)
        fig.colorbar(im,ax=ax,shrink=.75)
    save(fig,'08_adaptive_errors')

    fig,axes=plt.subplots(2,2,figsize=(10.2,6.5),layout='constrained')
    for method in ['FDM','BT']:
        for mode,marker,ls in [('fixed','o','-'),('adaptive','^','--')]:
            rows=[row for row in time_adapt if row['method']==method and row['mode']==mode]
            label=f'{method}, '+('fest' if mode=='fixed' else 'adaptiv')
            err=column(rows,'error_time_only')
            axes[0,0].loglog(column(rows,'seconds'),err,marker+ls,color=COLORS[method],label=label)
            flops=sum(column(rows,key) for key in ['solve_flops','factor_flops','rhs_flops','output_flops'])
            axes[0,1].loglog(flops,err,marker+ls,color=COLORS[method],label=label)
    axes[0,0].set(xlabel='Medianzeit [s]',ylabel='Reiner Zeitfehler im Ausgang',title='Weniger Schritte heißt nicht\nunbedingt schneller')
    axes[0,1].set(xlabel='Arithmetisches Strukturmodell [Operationen]',ylabel='Reiner Zeitfehler im Ausgang',title='Arithmetische Arbeit: LU, Lösen\nund Kontrollausgabe')
    rows=[row for row in time_adapt if row['mode']=='adaptive']
    x=np.arange(len(rows));accepted=column(rows,'accepted');rejected=column(rows,'rejected')
    axes[1,0].bar(x,accepted,color=GREY,label='Akzeptierte Makroschritte')
    axes[1,0].bar(x,rejected,bottom=accepted,color=ORANGE,label='Verworfene Versuche')
    axes[1,0].set_xticks(x,[f"{row['method']}\n{row['rtol']:.0e}" for row in rows],fontsize=11)
    axes[1,0].set(ylabel='Anzahl',title='Drei Lösevorgänge pro Zeitversuch')
    for method in ['FDM','BT']:
        data=np.load(DATA/f'adaptive_time_{method}.npz');attempts=data['attempts']
        good=attempts[:,3]==1
        axes[1,1].semilogy(attempts[good,0],attempts[good,1],'.-',ms=3,color=COLORS[method],label=method)
        axes[1,1].semilogy(attempts[~good,0],attempts[~good,1],'x',ms=5,color=COLORS[method])
    axes[1,1].set(xlabel='Zeit t',ylabel='Versuchter Makroschritt h',title=r'Schrittweiten bei rtol=$10^{-3}$; × verworfen')
    for ax in axes.flat:ax.legend()
    save(fig,'09_adaptive_time')

    # Include forcing and illustrate global coordinate directions.
    from helpers import balancing,fdm_2d_poisson
    A,B,*_=fdm_2d_poisson(17); Ab,Bb,Cb,sigma=balancing(A,B,np.eye(289),1e-14)
    fig,axes=plt.subplots(1,4,figsize=(10.2,2.9),layout='constrained')
    grid=np.linspace(0,1,19)
    for j,ax in enumerate(axes):
        values=np.pad(Cb[:,j].reshape(17,17),1); bound=np.max(np.abs(values))
        ax.pcolormesh(grid,grid,values,cmap='RdBu_r',vmin=-bound,vmax=bound,shading='nearest')
        ax.set(title=f'BT-Richtung {j+1}',xlabel=r'$x_1$',ylabel=r'$x_2$',aspect='equal');ax.grid(False)
    save(fig,'10_global_directions')

    # Tables are generated from data to avoid transcribing numerical results.
    def fmt(value):return f'{value:.3g}'
    lines=[]
    for row in adaptive:
        lines.append(f"{row['method']} & {fmt(row['tolerance'])} & {int(row['states'])} & {fmt(row['estimate'])} & {fmt(row['error'])} & {1000*row['adaptation']:.1f} & {1000*row['online']:.1f}" + " " + chr(92)*2)
    (HERE/'adaptive_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for row in time_adapt:
        label=(f"fest, {int(row['steps'])}" if row['mode']=='fixed' else f"adaptiv, {row['rtol']:.0e}")
        flops=sum(row[key] for key in ['solve_flops','factor_flops','rhs_flops','output_flops'])
        lines.append(f"{row['method']} & {label} & {1000*row['seconds']:.2f} & {fmt(row['error_time_only'])} & {int(row['accepted'])}/{int(row['rejected'])} & {int(row['solves'])} & {int(row['factorizations'])} & {flops/1e6:.2f}" + " " + chr(92)*2)
    (HERE/'time_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for row in timing:
        lines.append(f"{int(row['n'])} & {1000*row['assembly']:.2f} & {1000*row['bt_build']:.2f} & {1000*row['fdm_online']:.2f} & {1000*row['bt_online']:.2f} & {1000*row['bt_reconstruct']:.3f}" + " " + chr(92)*2)
    (HERE/'runtime_rows.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for row in adaptive:
        if row['tolerance'] not in [.002,.0005]:continue
        lines.append(f"{row['method']} & {fmt(row['tolerance'])} & {int(row['trial_simulations'])} & {int(row['linear_solves'])} & {int(row['factorizations'])} & {int(row['lyapunov_solves'])} & {int(row['scalar_state_updates'])}" + " " + chr(92)*2)
    (HERE/'work_rows.tex').write_text('\n'.join(lines)+'\n')
    (HERE/'results_macros.tex').write_text(
        f"\\newcommand{{\\BreakEven}}{{{breakeven:.1f}}}\n"
        f"\\newcommand{{\\ReferenceGap}}{{{ref_summary['reference_check']:.2g}}}\n"
        f"\\newcommand{{\\PDEFloor}}{{{ref_summary['pde_floor']:.3g}}}\n")
    print('Ten figure groups and measured tables generated')


if __name__=='__main__':main()
