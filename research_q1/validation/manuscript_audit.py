"""Reproduce additional exploratory evidence without modifying inherited results.

All calibration decisions exclude October labels. Reusing October nevertheless
makes this a development analysis, not a newly independent confirmation.
"""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research_q1.baselines.metadata_benchmark import load_period, design, type_rate, sha256
from research_q1.novel.conformal import finite_quantile
from research_q1.stats.inference import cluster_bootstrap_difference
from research_q1.validation.metrics import regression_metrics, interval_metrics

ROOT = Path(__file__).resolve().parents[2]


def duration_cell(frame):
    return frame.aircraft_type.astype(str) + '/' + pd.cut(
        frame.duration_min, [-np.inf, 5, 15, np.inf],
        labels=['short', 'medium', 'long'], right=False).astype(str)


def block_crc_threshold(scores, events, flights, beta, grid):
    """Bounded monotone loss averaged within flight; fixed threshold candidates.

    The empirical bound is for a random flight's mean joint event, never the
    conditional error rate among releases. No shift guarantee is asserted.
    """
    losses = []
    for threshold in grid:
        loss = pd.Series((scores <= threshold) * events).groupby(
            np.asarray(flights)).mean()
        losses.append((float(loss.sum()) + 1) / (len(loss) + 1))
    feasible = np.flatnonzero(np.asarray(losses) <= beta)
    if not len(feasible):
        return None, losses
    return float(grid[feasible[-1]]), losses


def inventory_metrics(y, prediction, flights, repetitions, seed):
    totals = pd.DataFrame({'y': y, 'p': prediction, 'flight': flights}).groupby('flight')[['y', 'p']].sum()
    values = totals.to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        sample = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        draws.append((sample[1] - sample[0]) / sample[0])
    return {'reference_fuel_kg': float(np.sum(y)), 'estimated_fuel_kg': float(np.sum(prediction)),
            'relative_bias': float((np.sum(prediction) - np.sum(y)) / np.sum(y)),
            'bias_flight_bootstrap_ci': np.quantile(draws, [.025, .975]).tolist(),
            'absolute_bias_bootstrap_95th': float(np.quantile(np.abs(draws), .95)),
            'flight_blocks': len(values)}


def run():
    started = time.perf_counter()
    config_path = ROOT / 'research_q1/configs/development_audit.json'
    cfg = json.loads(config_path.read_text())
    train, cal, test = [load_period(ROOT / 'data/raw', name) for name in ('train', 'rank', 'final')]
    ids = cal[['flight_id', 'takeoff']].drop_duplicates().sort_values(['takeoff', 'flight_id']).flight_id.to_numpy()
    groups = np.array_split(ids, 3)
    masks = [cal.flight_id.isin(g).to_numpy() for g in groups]
    assert not set(train.flight_id) & set(cal.flight_id)
    assert not set(train.flight_id) & set(test.flight_id)
    assert not set(cal.flight_id) & set(test.flight_id)
    assert sum(m.sum() for m in masks) == len(cal)
    evaluation = pd.concat([cal, test], ignore_index=True)
    x, xe, columns = design(train, evaluation)
    y, yc, yt = train.fuel_kg.to_numpy(), cal.fuel_kg.to_numpy(), test.fuel_kg.to_numpy()
    result = {'status': cfg['status'], 'config': cfg, 'config_sha256': sha256(config_path),
              'code_sha256': sha256(Path(__file__)),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'hardware': platform.platform() + ' ' + platform.machine(),
              'source_sha256': {p.name: sha256(p) for p in sorted((ROOT / 'data/raw').glob('*.parquet'))},
              'calibration_counts': [{'flights': len(g), 'intervals': int(m.sum())} for g, m in zip(groups, masks)],
              'point': {}, 'interval': {}, 'cells': [], 'selection': [], 'inventory': {}}
    predictions = {}
    for name in ('retrospective_hgb', 'landing_time_free_hgb'):
        keep = np.array([name == 'retrospective_hgb' or col not in
                         ('flight_duration_min', 'remaining_to_landing_min') for col in columns])
        tick = time.perf_counter()
        model = HistGradientBoostingRegressor(max_iter=cfg['iterations'], learning_rate=.05,
                 max_leaf_nodes=31, random_state=cfg['seed'], early_stopping=False).fit(x[:, keep], y)
        fit_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        predictions[name] = np.maximum(0, model.predict(xe[:, keep]))
        seconds = time.perf_counter() - tick
        result['point'][name] = {**regression_metrics(yt, predictions[name][len(cal):]),
             'fit_seconds': fit_seconds, 'predict_seconds': seconds,
             'prediction_rows': len(evaluation), 'rows_per_second': len(evaluation) / seconds,
             'features': [col for col, yes in zip(columns, keep) if yes]}
    predictions['type_median_rate'] = type_rate(train, evaluation)
    result['point']['type_median_rate'] = regression_metrics(yt, predictions['type_median_rate'][len(cal):])
    result['paired'] = {}
    ref = np.abs(predictions['retrospective_hgb'][len(cal):] - yt)
    for name in ('landing_time_free_hgb', 'type_median_rate'):
        result['paired'][name] = cluster_bootstrap_difference(ref,
            np.abs(predictions[name][len(cal):] - yt), test.route_id.to_numpy(),
            repetitions=cfg['bootstrap_repetitions'], seed=cfg['seed'])

    pc, pt = np.split(predictions['retrospective_hgb'], [len(cal)])
    error = np.abs(yc - pc)
    ac, at = duration_cell(cal).to_numpy(), duration_cell(test).to_numpy()
    a, b, c = masks
    alpha = cfg['alpha']
    global_a, global_b = finite_quantile(error[a], alpha), finite_quantile(error[b], alpha)
    # Type parents are shrunk toward global; duration cells toward their type.
    type_prior = {}
    for name in np.unique(cal.aircraft_type):
        member = a & (cal.aircraft_type.to_numpy() == name)
        n = member.sum()
        type_prior[name] = ((n * finite_quantile(error[member], alpha) + cfg['shrinkage_kappa'] * global_a)
                            / (n + cfg['shrinkage_kappa'])) if n >= 10 else global_a
    multipliers = {}
    hard = {}
    for cell in np.unique(np.concatenate([ac, at])):
        parent = type_prior.get(cell.split('/')[0], global_a)
        member = a & (ac == cell)
        n = member.sum()
        multipliers[cell] = ((n * finite_quantile(error[member], alpha) + cfg['shrinkage_kappa'] * parent)
                             / (n + cfg['shrinkage_kappa'])) if n >= 10 else parent
        member_b = b & (ac == cell)
        hard[cell] = finite_quantile(error[member_b], alpha) if member_b.sum() >= cfg['min_cell_intervals'] else global_b
    normalizer_cal = np.array([multipliers[g] for g in ac])
    correction = finite_quantile(error[b] / np.maximum(normalizer_cal[b], 1e-9), alpha)
    half_cal = correction * normalizer_cal
    half_test = correction * np.array([multipliers[g] for g in at])
    widths = {'split_B': np.full(len(test), global_b),
              'hard_mondrian_B': np.array([hard[g] for g in at]),
              'shrinkage_A_plus_global_B': half_test}
    result['stage_b_correction'] = correction
    for name, half in widths.items():
        lo, hi = np.maximum(0, pt-half), pt+half
        result['interval'][name] = interval_metrics(yt, lo, hi, alpha)
        eligible = []
        for cell in sorted(np.unique(at)):
            member = at == cell
            n = int(member.sum())
            covered = (yt[member] >= lo[member]) & (yt[member] <= hi[member])
            # Cluster bootstrap respects dependence within a flight; Wilson alone does not.
            from research_q1.stats.inference import cluster_bootstrap_mean
            flights = test.flight_id.to_numpy()[member]
            ci = cluster_bootstrap_mean(covered.astype(float), flights,
                  repetitions=cfg['bootstrap_repetitions'], seed=cfg['seed'])['cluster_bootstrap_95pct_ci'] if len(np.unique(flights)) >= 2 else None
            row = {'method': name, 'cell': cell, 'n': n, 'flights': len(np.unique(flights)),
                   'calibration_B_n': int((b & (ac == cell)).sum()),
                   'coverage': float(covered.mean()), 'coverage_flight_bootstrap_ci': ci,
                   'width_kg': float((hi-lo)[member].mean()),
                   'eligible': n >= cfg['min_cell_intervals'] and len(np.unique(flights)) >= 10}
            result['cells'].append(row)
            if row['eligible']: eligible.append(row)
        result['interval'][name]['eligible_cells'] = len(eligible)
        result['interval'][name]['worst_cell_coverage'] = min(row['coverage'] for row in eligible)

    score_c = half_cal[c] / np.maximum(pc[c], cfg['absolute_tolerance_kg'])
    score_t = half_test / np.maximum(pt, cfg['absolute_tolerance_kg'])
    event_c = error[c] > np.maximum(cfg['absolute_tolerance_kg'], cfg['relative_tolerance'] * pc[c])
    event_t = np.abs(yt-pt) > np.maximum(cfg['absolute_tolerance_kg'], cfg['relative_tolerance'] * pt)
    grid = np.concatenate([[-1], np.linspace(0, 5, 501), [1000000.]])
    threshold, corrected_losses = block_crc_threshold(score_c, event_c, cal.flight_id.to_numpy()[c], cfg['beta'], grid)
    result['crc'] = {'threshold': threshold, 'beta': cfg['beta'], 'calibration_flights': len(groups[2]),
                     'target': 'flight-mean JOINT release and large error',
                     'assumptions': 'exchangeability of flight blocks; not established under time shift'}
    curve_thresholds = sorted(set([.1, .2, .3, .5, .75, 1, 1.5, 2, 3, 5, 1000000.] + ([threshold] if threshold is not None else [])))
    for t in curve_thresholds:
        selected = score_t <= t
        joint_flight = pd.Series(selected * event_t).groupby(test.flight_id.to_numpy()).mean()
        result['selection'].append({'threshold': t, 'crc_selected': t == threshold,
            'release_fraction': float(selected.mean()), 'selected_n': int(selected.sum()),
            'selected_mae_kg': float(np.abs(pt-yt)[selected].mean()) if selected.any() else None,
            'joint_flight_loss': float(joint_flight.mean()),
            'released_error_fraction': float(event_t[selected].mean()) if selected.any() else None,
            'represented_reference_fuel_fraction': float(yt[selected].sum()/yt.sum()),
            'min_type_release_fraction': float(pd.Series(selected).groupby(test.aircraft_type.to_numpy()).mean().min())})
    z = score_t <= threshold if threshold is not None else np.zeros(len(test), bool)
    for name, pred in [('all_point', pt), ('released_only', np.where(z, pt, 0)),
                       ('released_plus_same_predictor', np.where(z, pt, pt)),
                       ('released_plus_type_median', np.where(z, pt, predictions['type_median_rate'][len(cal):]))]:
        result['inventory'][name] = inventory_metrics(yt, pred, test.flight_id.to_numpy(), cfg['bootstrap_repetitions'], cfg['seed'])
    assert result['inventory']['all_point'] == result['inventory']['released_plus_same_predictor']
    result['runtime_seconds'] = time.perf_counter() - started
    out = ROOT / 'research_q1/outputs/manuscript-audit.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('point', 'interval', 'crc', 'inventory', 'runtime_seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    run()
