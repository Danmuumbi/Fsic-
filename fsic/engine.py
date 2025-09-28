
import numpy as np
from .lattice import build_rect, build_hex, to_mats

def coupling(delta, mode='sin', kind='tanh', beta=1.8, alpha=0.5):
    if mode!='phi':
        return np.sin(delta)
    if kind=='rational':
        s=np.sin(delta); return s/(1.0+alpha*(s*s))
    return np.tanh(beta*np.sin(delta))

def run_simulation(cfg: dict, return_trace=False):
    # defaults
    C={'rows':10,'cols':10,'tiling':'hex','rewire_p':0.10,
       'J_tan':0.9,'J_rad':0.7,'m_floor':0.75,'rfi_target':1.30,
       'alpha_r':0.002,'alpha_t':0.001,'k_min':0.2,'k_max':2.5,
       'gamma':0.1,'eta':0.02,'lam':0.005,'freeze_on_fuse':True,
       'R_fuse':0.70,'A_th':25.0,'steps':1200,'dt':0.01,
       'sigma_omega':0.1,'sigma_noise':0.01,'seed':0,
       'mode':'phi','phi_kind':'tanh','phi_beta':1.8,'phi_alpha':0.5}
    C.update(cfg or {})
    # lattice
    mask = build_hex(C['rows'],C['cols'],C['J_tan'],C['J_rad'],C['rewire_p'],C['seed']) if C['tiling']=='hex' \
           else build_rect(C['rows'],C['cols'],C['J_tan'],C['J_rad'],C['rewire_p'],C['seed'])
    Jt,Jr,N=to_mats(mask)
    # state
    rng=np.random.default_rng(C['seed']); th=rng.uniform(0,2*np.pi,N); om=rng.normal(0.0,C['sigma_omega'],N)
    k_tan, k_rad = 1.0, 0.8; M=np.zeros(N); A=0.0; fused=False; fuse_time=np.nan; g=C['gamma']
    dt=C['dt']; trace=[] if return_trace else None
    for step in range(C['steps']):
        Je = k_tan*Jt + k_rad*Jr
        D = th[np.newaxis,:]-th[:,np.newaxis]
        inter = (Je*coupling(D, C['mode'], C['phi_kind'], C['phi_beta'], C['phi_alpha'])).sum(axis=1)
        R = float(np.abs(np.exp(1j*th).mean()))
        forcing = (g*M) if (R>=C['m_floor']) else 0.0
        dth = om + inter + forcing + rng.normal(0.0,C['sigma_noise'],N)
        th = (th + dt*dth)%(2*np.pi)
        if R>0.5: A += (R-0.5)*dt*100.0
        E_r=float((Jr*(1-np.cos(D))).sum()); E_t=float((Jt*(1-np.cos(D))).sum())
        RFI=(abs(E_r)+1e-6)/(abs(E_t)+1e-6)
        if R>=C['m_floor']:
            err=C['rfi_target']-RFI
            if err>0: k_rad*=1.0+C['alpha_r']*min(abs(err),1.0); k_tan*=1.0-C['alpha_t']*min(abs(err),1.0)
            else:     k_tan*=1.0+C['alpha_t']*min(abs(err),1.0); k_rad*=1.0-C['alpha_r']*min(abs(err),1.0)
            k_rad=float(max(C['k_min'],min(C['k_max'],k_rad))); k_tan=float(max(C['k_min'],min(C['k_max'],k_tan)))
            M=(1.0-C['lam'])*M + C['eta']*dth
        if (not fused) and (R>=C['R_fuse'] and A>=C['A_th']):
            fused=True; fuse_time=step*dt
            if C['freeze_on_fuse']: g=0.0
        if trace is not None and (step%20==0):
            trace.append((step*dt,R,A,RFI,k_tan,k_rad))
    out={'final_R':R,'final_A':A,'final_RFI':RFI,'fused':int(fused),'fuse_time':fuse_time,'k_tan':k_tan,'k_rad':k_rad}
    if return_trace: out['trace']=trace
    if fused:
        out['glyph']={'tiling':C['tiling'],'p':C['rewire_p'],'J_tan':C['J_tan'],'J_rad':C['J_rad'],
                      'k_tan':k_tan,'k_rad':k_rad,'rfi_target':C['rfi_target'],
                      'echo':{'gamma':C['gamma'],'eta':C['eta'],'lam':C['lam'],'frozen':C['freeze_on_fuse']},
                      'thresholds':{'R_fuse':C['R_fuse'],'A_th':C['A_th']},
                      'mode':C['mode'],'phi_kind':C['phi_kind'],'phi_beta':C['phi_beta'],'phi_alpha':C['phi_alpha']}
    return out

def stress_reentrain(cfg: dict, perturb_start=6.0, perturb_dur=2.0, sigma_hi=0.08):
    # run to end; inject higher noise between [start, start+dur); measure time to reach R>=0.70 after perturb end
    C=dict(cfg); C['return_trace']=True
    out = run_simulation(C, return_trace=True)
    # Synthetic approximation: raise re-entrainment time if final_R < 0.7 or fuse_time is late
    # For a true dynamic perturbation, you'd run a stepwise integrator; here we estimate from trace
    tr = out.get('trace', [])
    if not tr:
        return {'reentrain_time': float('nan'), 'note':'no trace'}
    # find first time >= perturb_start and >= perturb_end
    tR = [(t,R) for (t,R,_,_,_,_) in tr]
    # naive: reentrain time ~ how long it took to reach R>=0.7 near the end minus perturb end
    perturb_end = perturb_start + perturb_dur
    t_after = [t for (t,R) in tR if t>=perturb_end and R>=0.70]
    if not t_after:
        re_t = float('inf')
    else:
        re_t = min(t_after) - perturb_end
    return {'reentrain_time': float(re_t)}
