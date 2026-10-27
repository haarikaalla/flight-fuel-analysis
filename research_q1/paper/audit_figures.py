"""Vector and raster scientific figures for the manuscript audit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_q1/paper/revision/figures'

def run():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads((ROOT/'research_q1/outputs/manuscript-audit.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(name):
        plt.tight_layout()
        plt.savefig(OUT/f'{name}.pdf', bbox_inches='tight', metadata={'CreationDate': None})
        plt.savefig(OUT/f'{name}.png', dpi=220, bbox_inches='tight')
        plt.close()
    cells = [r for r in d['cells'] if r['method']=='shrinkage_A_plus_global_B' and r['eligible']]
    cells.sort(key=lambda r:r['coverage'])
    cells = cells[:12]
    plt.figure(figsize=(6.2,4))
    for index,row in enumerate(cells):
        low,high=row['coverage_flight_bootstrap_ci']
        plt.errorbar(100*row['coverage'],index,xerr=[[100*(row['coverage']-low)],[100*(high-row['coverage'])]],fmt='o',color='#215779',capsize=3)
    plt.yticks(range(len(cells)),[r['cell'] for r in cells])
    plt.axvline(90,color='#ac3333',linestyle='--',label='Nominal 90%')
    plt.xlabel('Coverage (%) with 95% flight-bootstrap interval')
    plt.legend(loc='upper left',fontsize=8)
    save('cell_coverage')
    rows=[r for r in d['selection'] if r['selected_n']]
    fig,ax=plt.subplots(1,2,figsize=(7,3))
    x=[100*r['release_fraction'] for r in rows]
    ax[0].plot(x,[r['selected_mae_kg'] for r in rows],'o-',color='#215779')
    ax[0].set_ylabel('Released MAE (kg)')
    ax[1].plot(x,[100*r['joint_flight_loss'] for r in rows],'o-',label='Joint loss per flight')
    ax[1].plot(x,[100*r['released_error_fraction'] for r in rows],'s-',label='Conditional on release')
    ax[1].axhline(5,color='gray',linestyle='--')
    ax[1].set_ylabel('Large error (%)')
    ax[1].legend(fontsize=7)
    for a in ax:a.set_xlabel('Released intervals (%)')
    save('release_curve')
    rows=list(d['inventory'].items())
    plt.figure(figsize=(6.2,3))
    labels=['All predictions','Released only','Same-model imputation','Type-median imputation']
    for i,(name,row) in enumerate(rows):
        # JSON keys sorted; retain labels through explicit mapping.
        label={'all_point':labels[0],'released_only':labels[1],'released_plus_same_predictor':labels[2],'released_plus_type_median':labels[3]}[name]
        low,high=row['bias_flight_bootstrap_ci']; val=row['relative_bias']
        plt.errorbar(val*100,label,xerr=[[100*(val-low)],[100*(high-val)]],fmt='o',capsize=4,color='#215779')
    plt.axvline(0,color='gray',linestyle='--')
    plt.xlabel('Observed-interval inventory relative bias (%)')
    save('inventory_bias')
    plt.figure(figsize=(6.2,3))
    for name,label in [('split_B','Split B'),('hard_mondrian_B','Hard Mondrian B'),('shrinkage_A_plus_global_B','Shrinkage A + B')]:
        r=d['interval'][name]
        plt.scatter(r['mean_width_kg'],100*r['coverage'],s=65,label=label)
    plt.axhline(90,color='gray',linestyle='--')
    plt.xlabel('Mean width (kg)');plt.ylabel('Marginal coverage (%)');plt.legend(fontsize=8)
    save('calibration_tradeoff')

if __name__=='__main__':run()
