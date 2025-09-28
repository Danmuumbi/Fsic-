
import json, os, csv, math, statistics
from fsic import run_simulation, stress_reentrain

GOLD = json.load(open(os.path.join('presets','gold_phi_tanh_1p8.json')))
os.makedirs('results', exist_ok=True)

def repeatability():
    rows=[]
    for mode in ['sin','phi']:
        cfg=dict(GOLD); cfg['mode']=mode
        for seed in [1,2,3,4,5]:
            cfg['seed']=seed
            out=run_simulation(cfg, return_trace=False)
            rows.append({'mode':mode,'seed':seed,'fused':out['fused'],
                         'fuse_time':out['fuse_time'],'final_R':out['final_R']})
    with open('results/repeatability.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    # summary
    def summ(mode):
        sub=[r for r in rows if r['mode']==mode]
        fr=sum(r['fused'] for r in sub)/len(sub)
        mt=statistics.median([r['fuse_time'] for r in sub if r['fused']]) if any(r['fused'] for r in sub) else float('nan')
        mR=sum(r['final_R'] for r in sub)/len(sub)
        return fr, mt, mR
    sin = summ('sin'); phi = summ('phi')
    return {'repeatability': {'sin':sin, 'phi':phi}}

def transfer():
    # baseline glyph (seed 7)
    base=dict(GOLD); base['seed']=7
    outA=run_simulation(base, return_trace=False)
    # transfer: change seed & noise a bit
    tcfg=dict(GOLD); tcfg['seed']=23; tcfg['sigma_noise']=0.012
    outB=run_simulation(tcfg, return_trace=False)
    rows=[{'case':'A_base','fused':outA['fused'],'fuse_time':outA['fuse_time'],'final_R':outA['final_R']},
          {'case':'B_transfer','fused':outB['fused'],'fuse_time':outB['fuse_time'],'final_R':outB['final_R']}]
    with open('results/transfer.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return {'transfer': rows}

def stress():
    cfg=dict(GOLD); cfg['seed']=11
    met=stress_reentrain(cfg, perturb_start=6.0, perturb_dur=2.0, sigma_hi=0.08)
    with open('results/stress.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['reentrain_time']); w.writeheader(); w.writerow(met)
    return {'stress': met}

if __name__=='__main__':
    rep = repeatability()
    trf = transfer()
    sts = stress()
    # write summary
    lines=[]
    fr_sin, mt_sin, mR_sin = rep['repeatability']['sin']
    fr_phi, mt_phi, mR_phi = rep['repeatability']['phi']
    lines.append(f"Repeatability — sin: fuse_rate={fr_sin:.2f}, median_fuse_time={mt_sin:.2f}, mean_R={mR_sin:.3f}")
    lines.append(f"Repeatability — phi: fuse_rate={fr_phi:.2f}, median_fuse_time={mt_phi:.2f}, mean_R={mR_phi:.3f}")
    lines.append(f"Transfer — A vs B: see results/transfer.csv (expect ±15% on fuse_time and final_R)")
    lines.append(f"Stress — re‑entrainment time: {sts['stress']['reentrain_time']:.2f} s (target < 2.0 s)")
    # open('results/FSIC_Summary.txt','w').write("\n".join(lines))
    open('results/FSIC_Summary.txt', 'w', encoding='utf-8').write("\n".join(lines))

    print("\n".join(lines))
