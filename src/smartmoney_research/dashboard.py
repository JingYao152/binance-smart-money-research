"""Local DuckDB research views. Loading and rendering never fetch source data."""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd

from smartmoney_research.analytics.evidence import build_evidence_bundle
from smartmoney_research.analytics.features import align_trade_events, compute_indicators
from smartmoney_research.quality import check_integrity
from smartmoney_research.storage import ResearchStore


def _scoped_rows(store, table, trader_id):
    return [row for row in store.rows(table) if row.get('trader_id') == trader_id]


def local_bars(store, symbol, interval):
    """Return verified final bars and retain excluded-row coverage in attrs."""
    rows = store.connection.execute(
        '''SELECT *,
           fetched_at IS NOT NULL AND close_time IS NOT NULL
             AND close_time < epoch_ms(fetched_at)
             AND close_time < epoch_ms(current_timestamp) AS __verified_final,
           fetched_at IS NULL AS __missing_retrieval,
           close_time >= epoch_ms(current_timestamp) AS __future_close,
           fetched_at IS NOT NULL AND close_time >= epoch_ms(fetched_at) AS __retrieved_before_close,
           close_time IS NULL AS __missing_close_time
           FROM klines WHERE symbol = ? AND interval = ? ORDER BY open_time''',
        [symbol, interval],
    ).fetchdf()
    available = rows['__verified_final'].fillna(False)
    coverage = {'stored_rows': len(rows), 'verified_rows': int(available.sum()),
                'unavailable_rows': int((~available).sum())}
    reasons = ('missing_retrieval', 'future_close', 'retrieved_before_close', 'missing_close_time')
    coverage.update({reason: int(rows[f'__{reason}'].fillna(False).sum()) for reason in reasons})
    bars = rows.loc[available].drop(columns=['__verified_final', *[f'__{reason}' for reason in reasons]])
    bars.attrs.update(symbol=symbol, coverage=coverage)
    return bars


def analyze_local(store, trader_id, *, interval='1h'):
    """Join source events to strictly earlier closed bars; preserve missingness."""
    bundle = build_evidence_bundle(store, trader_id).to_dict()
    events = []
    for table in ('position_history', 'open_positions', 'order_history'):
        for row in _scoped_rows(store, table, trader_id):
            times = (('order', 'timestamp'),) if table == 'order_history' else (('open', 'open_time'), ('close', 'close_time'))
            for kind, field in times:
                if row.get(field) is not None:
                    events.append(dict(table=table, record_id=row['record_id'],
                                       market_type=row['market_type'], symbol=row['symbol'],
                                       event_kind=kind, event_time=row[field]))
    result = []
    market_coverage = {}
    for symbol in sorted({event['symbol'] for event in events}, key=lambda item: item or ''):
        scoped = [event for event in events if event['symbol'] == symbol]
        bars = local_bars(store, symbol, interval) if symbol else pd.DataFrame()
        if symbol:
            market_coverage[symbol] = bars.attrs['coverage']
        if bars.empty:
            status = 'unavailable_market_bars' if bars.attrs.get('coverage', {}).get('stored_rows') else 'missing_market_bars'
            result.extend({**event, 'feature_status': status, 'bar_close_time': None} for event in scoped)
            continue
        features = compute_indicators(bars)
        first_close = features['close_time'].min()
        eligible = [event for event in scoped if event['event_time'] > first_close]
        result.extend({**event, 'feature_status': 'no_prior_closed_bar', 'bar_close_time': None}
                      for event in scoped if event['event_time'] <= first_close)
        if eligible:
            aligned = align_trade_events(pd.DataFrame(eligible), features)
            result.extend({**row, 'feature_status': 'aligned', 'label': 'derived'}
                          for row in aligned.to_dict('records'))
    result.sort(key=lambda event: event['event_time'])
    return {'evidence': bundle, 'interval': interval, 'trade_events': result,
            'market_bar_coverage': market_coverage,
            'feature_alignment': 'bar close_time strictly earlier than event_time; warm-up values remain missing'}


def load_dashboard_data(database, trader_id=None, *, interval='1h'):
    """Load a snapshot through a read-only DuckDB connection, without creation."""
    empty = dict(status='missing_database', traders=[], trader_id=None, profiles=[],
                 positions=[], charts=[], coverage={}, analysis={})
    if not Path(database).is_file():
        return empty
    with ResearchStore(database, read_only=True) as store:
        # An empty DuckDB file is also a valid empty dashboard input.
        tables = {row[0] for row in store.connection.execute('SHOW TABLES').fetchall()}
        if 'traders' not in tables:
            return {**empty, 'status': 'empty_database'}
        traders = sorted({row['trader_id'] for table in ('traders', 'profiles', 'position_history', 'open_positions', 'order_history', 'sync_state')
                          for row in store.rows(table)} |
                         {str(row[0]) for row in store.connection.execute(
                             "SELECT DISTINCT coalesce(json_extract_string(params, '$.traderId'), json_extract_string(params, '$.trader_id')) FROM raw_responses"
                         ).fetchall() if row[0] is not None})
        traders = [identity for identity in traders if not identity.startswith('__')]
        selected = trader_id or (traders[0] if traders else None)
        if selected is None:
            return {**empty, 'status': 'empty', 'traders': traders}
        analysis = analyze_local(store, selected, interval=interval)
        coverage = check_integrity(store, selected)
        coverage['market_bars'] = analysis['market_bar_coverage']
        return dict(status='ready', traders=traders, trader_id=selected,
                    profiles=_scoped_rows(store, 'profiles', selected),
                    positions=_scoped_rows(store, 'position_history', selected) + _scoped_rows(store, 'open_positions', selected),
                    charts=_scoped_rows(store, 'charts', selected),
                    coverage=coverage, analysis=analysis)


def validate_backtest_report(report):
    """Validate every nested field consumed by the renderer before use."""
    if not isinstance(report, dict) or not isinstance(report.get('symbols'), dict):
        raise ValueError('expected a backtest report object with symbols')
    if not isinstance(report.get('trader_id'), str) or not isinstance(report.get('rule', {}), dict):
        raise ValueError('expected a trader_id string and rule object')
    for symbol, run in report['symbols'].items():
        if not isinstance(run, dict):
            raise ValueError(f'{symbol}: expected a result object')
        single = run.get('single')
        if not isinstance(single, dict) or any(
            not isinstance(single.get(field), dict)
            for field in ('metrics', 'assumptions', 'benchmark', 'funding', 'liquidation_risk')
        ):
            raise ValueError(f'{symbol}: incomplete single-run report')
        if {'symbol', 'rule'} & single['metrics'].keys():
            raise ValueError(f'{symbol}: metrics contain reserved comparison fields')
        if not isinstance(run.get('walk_forward'), list) or not isinstance(run.get('sensitivity'), dict):
            raise ValueError(f'{symbol}: expected walk_forward list and sensitivity object')
    return report


def render_dashboard(ui, data, backtest_reports=()):
    """Render a supplied local snapshot; no client or network dependency."""
    ui.title('Local smart-money research')
    ui.caption('Observed provider snapshots; derived indicators; estimated backtest risk. Sampled PnL is not account equity.')
    if data['status'] != 'ready':
        ui.info('No local trader records. Run sync separately, then reload this page.')
        return
    ui.subheader('Profile snapshots · observed')
    ui.dataframe(pd.DataFrame(data['profiles']))
    ui.subheader('Source coverage and missing fields')
    ui.json(data['coverage'])
    ui.subheader('Positions · observed')
    ui.dataframe(pd.DataFrame(data['positions']))
    ui.subheader('Sampled PnL chart · observed provider samples')
    charts = pd.DataFrame(data['charts'])
    if charts.empty:
        ui.info('No local chart samples.')
    else:
        # Keep periods and chart types separate: ROI must never be drawn as PnL.
        for (period, chart_type), points in charts.groupby(['period', 'chart_type'], dropna=False):
            ui.caption(f'{period} / {chart_type} · sampled provider values')
            series = points[['timestamp', 'value']].dropna().sort_values('timestamp')
            series['timestamp'] = pd.to_datetime(series['timestamp'], unit='ms', utc=True)
            ui.line_chart(series.set_index('timestamp'))
    ui.subheader('Per-event market indicators · derived, strict as-of')
    ui.dataframe(pd.DataFrame(data['analysis']['trade_events']))
    ui.subheader('Local backtest comparisons · derived; liquidation risk is estimated')
    comparisons = []
    for report in backtest_reports:
        try:
            validate_backtest_report(report)
        except ValueError as exc:
            ui.error(f'Cannot read report: {exc}')
            continue
        if report.get('trader_id') != data['trader_id']:
            ui.info('A report for another trader was skipped.')
            continue
        for symbol, run in report.get('symbols', {}).items():
            if run.get('single'):
                comparisons.append(dict(symbol=symbol, rule=report.get('rule', {}).get('kind'), **run['single']['metrics']))
                ui.json({'symbol': symbol, 'assumptions': run['single']['assumptions'],
                         'benchmark': run['single']['benchmark'], 'funding': run['single']['funding'],
                         'liquidation_risk': run['single']['liquidation_risk'],
                         'walk_forward': run['walk_forward'], 'sensitivity': run['sensitivity']})
    if comparisons:
        ui.dataframe(pd.DataFrame(comparisons))
    else:
        ui.info('Upload local backtest JSON reports to compare results.')


def run_dashboard(database):
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Install the dashboard extra: python -m pip install -e ".[dashboard]"') from exc
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(Path(__file__).resolve()),
                    '--server.address=127.0.0.1', '--browser.gatherUsageStats=false',
                    '--', str(Path(database).resolve())], check=True)


def _app():
    import streamlit as st
    database = sys.argv[1]
    data = load_dashboard_data(database)
    if data['traders']:
        trader_id = st.selectbox('Local trader', data['traders'])
        if trader_id != data['trader_id']:
            data = load_dashboard_data(database, trader_id)
    files = st.file_uploader('Local backtest JSON reports', type=['json'], accept_multiple_files=True)
    reports = []
    for file in files or []:
        try:
            report = json.load(file)
            reports.append(validate_backtest_report(report))
        except (ValueError, TypeError) as exc:
            st.error(f'Cannot read report: {exc}')
    render_dashboard(st, data, reports)


if __name__ == '__main__':
    _app()
