import os
import json
import calendar
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf

START = '2005-01-01'
NY = ZoneInfo('America/New_York')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

FRED_SERIES = [
    'ICSA','CCSA','UMCSENT','NFCI','T10Y3M','T10YIE','T5YIE','T5YIFR','CPIAUCSL','DFF',
    'DTWEXBGS','DFII10','T10Y2Y','DGS2','DGS3MO','BAA10YM','VIXCLS','RRPONTSYD','WTREGEN','WALCL'
]
YF_SYMBOLS = ['SPY','RSP','GLD','HG=F','^VIX3M','^VXN']

LABELS = {
    'ICSA': 'Initial Claims',
    'CCSA': 'Continuing Claims',
    'UMCSENT': 'Consumer Sentiment',
    'NFCI': 'Chicago Fed NFCI',
    'T10Y3M': '10Y-3M curve',
    'RSP_SPY': 'Breadth proxy RSP/SPY',
    'COPPER_GOLD': 'Copper / Gold',
    'T10YIE': '10Y breakeven inflation',
    'T5YIE': '5Y breakeven inflation',
    'T5YIFR': '5y5y forward inflation',
    'CPIAUCSL': 'CPI YoY anchor score',
    'DFII10': 'US 10Y real yield',
    'T10Y2Y': '10Y-2Y curve',
    'DGS2': '2Y minus Fed Funds proxy',
    'NET_LIQ': 'Net liquidity',
    'BAA10YM': 'BAA spread vs 10Y',
    'VIXCLS': 'VIX',
    'V_RATIO': 'VIX / VIX3M',
    'VXN_VIX': 'VXN - VIX'
}

WHY = {
    'ICSA': 'Fast labor-market stress gauge. Lower is better for risk assets.',
    'CCSA': 'Confirms whether labor weakness is becoming persistent.',
    'UMCSENT': 'Consumer confidence proxy and cyclical demand signal.',
    'NFCI': 'Broad financial conditions measure. Easier conditions support risk.',
    'T10Y3M': 'Cycle / recession signal via the yield curve slope.',
    'RSP_SPY': 'Internal market breadth proxy using equal-weight vs cap-weight.',
    'COPPER_GOLD': 'Cyclical growth versus safe-haven demand proxy.',
    'T10YIE': 'Checks whether long-term inflation expectations remain anchored.',
    'T5YIE': 'More reactive market inflation expectations measure.',
    'T5YIFR': 'Long-term inflation anchor quality measure.',
    'CPIAUCSL': 'Realized inflation distance from a 2.25% comfort zone.',
    'DFII10': 'Higher real yields pressure equity valuations.',
    'T10Y2Y': 'Additional cycle signal from the yield curve.',
    'DGS2': 'Proxy for expected policy path versus current policy rate.',
    'NET_LIQ': 'Approximation of system liquidity support.',
    'BAA10YM': 'Long-history credit stress proxy.',
    'VIXCLS': 'Implied equity fear gauge on the S&P 500.',
    'V_RATIO': 'Short-term volatility stress versus 3-month volatility.',
    'VXN_VIX': 'Relative stress in Nasdaq vs broad equity market.'
}

def fred_csv(series):
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    df = pd.read_csv(url)
    df.columns = ['DATE', series]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df[series] = pd.to_numeric(df[series], errors='coerce')
    return df.set_index('DATE')

def get_fred_monthly_frame():
    frames = [fred_csv(s) for s in FRED_SERIES]
    return pd.concat(frames, axis=1).sort_index()

def get_yf_monthly_frame():
    out = {}
    for s in YF_SYMBOLS:
        df = yf.download(s, start=START, auto_adjust=True, progress=False)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        out[s] = close.rename(s)
    return pd.concat(out.values(), axis=1).sort_index()

def expanding_percentile(s, min_obs=24):
    vals = s.astype(float)
    out = pd.Series(index=vals.index, dtype=float)
    arr = vals.values
    for i in range(len(arr)):
        hist = arr[: i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) < min_obs or np.isnan(arr[i]):
            out.iloc[i] = np.nan
        else:
            out.iloc[i] = 100.0 * (hist <= arr[i]).mean()
    return out

def transform_indicator(name, monthly):
    s = monthly[name] if name in monthly.columns else None
    if name in ['ICSA', 'CCSA']:
        return -np.log(s)
    if name == 'UMCSENT':
        return s
    if name == 'NFCI':
        return -s
    if name == 'T10Y3M':
        return s
    if name in ['T10YIE', 'T5YIE', 'T5YIFR']:
        return -np.abs(s - 2.25)
    if name == 'CPIAUCSL':
        yoy = s.pct_change(12) * 100
        return -np.abs(yoy - 2.25)
    if name == 'DTWEXBGS':
        return -(s.pct_change(6) * 100)
    if name == 'DFII10':
        return -s
    if name == 'T10Y2Y':
        return s
    if name == 'DGS2':
        return -(s - monthly['DFF'])
    if name == 'BAA10YM':
        return -s
    if name == 'VIXCLS':
        return -np.log(s)
    if name == 'RSP_SPY':
        rel = monthly['RSP'] / monthly['SPY']
        return rel.pct_change(3) * 100
    if name == 'COPPER_GOLD':
        rel = monthly['HG=F'] / monthly['GLD']
        return rel.pct_change(3) * 100
    if name == 'V_RATIO':
        rel = monthly['VIXCLS'] / monthly['^VIX3M']
        return -rel
    if name == 'VXN_VIX':
        spread = monthly['^VXN'] - monthly['VIXCLS']
        return -spread
    if name == 'NET_LIQ':
        net = monthly['WALCL'] - monthly['RRPONTSYD'] - monthly['WTREGEN']
        return net.pct_change(3) * 100
    raise KeyError(name)

def is_last_day_of_month(dt):
    return dt.day == calendar.monthrange(dt.year, dt.month)[1]

def next_month_end(dt):
    if dt.month == 12:
        y, m = dt.year + 1, 1
    else:
        y, m = dt.year, dt.month + 1
    return datetime(y, m, calendar.monthrange(y, m)[1], tzinfo=NY).date()

def build_payload(now_date):
    fred = get_fred_monthly_frame()
    ydf = get_yf_monthly_frame()
    daily = fred.join(ydf, how='outer').sort_index().ffill()
    monthly = daily.resample('ME').last()

    ind_names = [
        'ICSA','CCSA','UMCSENT','NFCI','T10Y3M','RSP_SPY','COPPER_GOLD',
        'T10YIE','T5YIE','T5YIFR','CPIAUCSL','DFII10','T10Y2Y','DGS2','NET_LIQ',
        'BAA10YM','VIXCLS','V_RATIO','VXN_VIX'
    ]

    transformed = {n: transform_indicator(n, monthly) for n in ind_names}
    tdf = pd.DataFrame(transformed)
    scores = pd.DataFrame(index=tdf.index)
    for c in tdf.columns:
        scores[c] = expanding_percentile(tdf[c], min_obs=24)

    pillars = pd.DataFrame(index=scores.index)
    pillars['growth'] = scores[['ICSA','CCSA','UMCSENT','NFCI','T10Y3M','RSP_SPY','COPPER_GOLD']].mean(axis=1)
    pillars['inflation'] = scores[['T10YIE','T5YIE','T5YIFR','CPIAUCSL']].mean(axis=1)
    pillars['rates'] = scores[['DFII10','T10Y2Y','DGS2','NET_LIQ']].mean(axis=1)
    pillars['credit_vol'] = scores[['BAA10YM','VIXCLS','V_RATIO','VXN_VIX']].mean(axis=1)
    pillars['composite'] = 0.30 * pillars['growth'] + 0.10 * pillars['inflation'] + 0.25 * pillars['rates'] + 0.35 * pillars['credit_vol']

    monthly['ma10'] = monthly['SPY'].rolling(10).mean()
    monthly['trend'] = (monthly['SPY'] > monthly['ma10']).astype(float)
    data = monthly.join(pillars)

    state = []
    prev = 0.0
    for _, row in data.iterrows():
        comp = row['composite']
        cred = row['credit_vol']
        tr = row['trend']
        if np.isnan(comp) or np.isnan(cred):
            prev = 0.0
        else:
            if (comp >= 48 and cred >= 45) or (tr == 1 and comp >= 45 and cred >= 35):
                prev = 1.0
            elif (comp <= 40 and tr == 0) or (cred <= 35 and tr == 0):
                prev = 0.0
        state.append(prev)
    data['regime_expo'] = state

    usable = data.dropna(subset=['composite', 'credit_vol'])
    latest = usable.iloc[-1]
    latest_date = usable.index[-1].date()
    next_calc = now_date if is_last_day_of_month(now_date) else next_month_end(now_date)

    signal = 'RISK ON' if float(latest['regime_expo']) >= 1 else 'RISK OFF'
    action = 'Acheter / conserver SPY' if signal == 'RISK ON' else 'Vendre / rester cash'

    indicator_rows = []
    for ind in ind_names:
        if ind == 'RSP_SPY':
            raw_val = (monthly['RSP'] / monthly['SPY']).iloc[-1]
        elif ind == 'COPPER_GOLD':
            raw_val = (monthly['HG=F'] / monthly['GLD']).iloc[-1]
        elif ind == 'V_RATIO':
            raw_val = (monthly['VIXCLS'] / monthly['^VIX3M']).iloc[-1]
        elif ind == 'VXN_VIX':
            raw_val = (monthly['^VXN'] - monthly['VIXCLS']).iloc[-1]
        elif ind == 'NET_LIQ':
            raw_val = (monthly['WALCL'] - monthly['RRPONTSYD'] - monthly['WTREGEN']).iloc[-1]
        else:
            raw_val = monthly[ind].iloc[-1]
        indicator_rows.append({
            'code': ind,
            'label': LABELS.get(ind, ind),
            'why': WHY.get(ind, ''),
            'score': None if pd.isna(scores[ind].iloc[-1]) else round(float(scores[ind].iloc[-1]), 2),
            'raw_value': None if pd.isna(raw_val) else round(float(raw_val), 4),
        })

    return {
        'generated_at_new_york': datetime.now(NY).isoformat(),
        'as_of_signal_date': str(latest_date),
        'current_signal': signal,
        'decision': action,
        'next_calculation_date': str(next_calc),
        'method': 'Calcul mensuel à la clôture du dernier jour ouvré du mois ; décision appliquée au prochain open.',
        'pillars': {
            'growth': round(float(latest['growth']), 2),
            'inflation': round(float(latest['inflation']), 2),
            'rates': round(float(latest['rates']), 2),
            'credit_vol': round(float(latest['credit_vol']), 2),
            'composite': round(float(latest['composite']), 2),
            'trend': 'positive' if float(latest['trend']) > 0 else 'negative',
            'exposure': int(float(latest['regime_expo']))
        },
        'rules': [
            'Risk-On si Composite >= 48 et Credit/Vol >= 45',
            'Risk-On si tendance positive et Composite >= 45 et Credit/Vol >= 35',
            'Risk-Off si Composite <= 40 et tendance négative',
            'Risk-Off si Credit/Vol <= 35 et tendance négative',
            'Sinon : conservation de l’état précédent'
        ],
        'indicators': indicator_rows
    }

def pill_color(v):
    if v >= 60:
        return '#0f766e'
    if v >= 45:
        return '#b45309'
    return '#b91c1c'

def render_dashboard(payload):
    rows_html = ''.join(
        f"<tr><td>{r['label']}</td><td>{r['raw_value']}</td><td>{r['score']}</td><td>{r['why']}</td></tr>" for r in payload['indicators']
    )
    signal = payload['current_signal']
    return f"""<!doctype html>
<html lang='fr'>
<head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Macro Engine Dashboard</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin:0; background:#f4f7fb; color:#1f2937; }}
.wrapper {{ max-width:1200px; margin:0 auto; padding:24px; }}
.hero {{ background:linear-gradient(135deg,#122B49,#1F3864); color:white; padding:28px; border-radius:18px; box-shadow:0 12px 32px rgba(18,43,73,.22); }}
.hero h1 {{ margin:0 0 8px 0; font-size:30px; }}
.hero p {{ margin:6px 0; color:#dbe7f7; }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-top:18px; }}
.card {{ background:white; border-radius:16px; padding:18px; box-shadow:0 8px 22px rgba(15,23,42,.08); }}
.card h3 {{ margin:0 0 8px 0; font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:#64748b; }}
.big {{ font-size:28px; font-weight:800; }}
.signal-on {{ color:#15803d; }}
.signal-off {{ color:#b91c1c; }}
.pillars {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:18px; }}
.pill {{ border-radius:14px; padding:16px; color:white; }}
.pill small {{ display:block; opacity:.85; margin-bottom:6px; }}
.rules, .tablebox {{ background:white; border-radius:16px; padding:18px; box-shadow:0 8px 22px rgba(15,23,42,.08); margin-top:18px; }}
ul {{ margin:8px 0 0 18px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:top; }}
th {{ background:#122B49; color:white; position:sticky; top:0; }}
.footer {{ color:#64748b; font-size:12px; margin-top:16px; }}
@media (max-width: 900px) {{ .grid, .pillars {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width: 640px) {{ .grid, .pillars {{ grid-template-columns:1fr; }} .wrapper{{padding:14px;}} }}
</style>
</head>
<body>
<div class='wrapper'>
  <div class='hero'>
    <h1>Macro Engine Dashboard — SPY</h1>
    <p>Signal de marché mensuel basé sur le modèle retenu.</p>
    <p><strong>Date du dernier signal :</strong> {payload['as_of_signal_date']} &nbsp;|&nbsp; <strong>Prochain calcul :</strong> {payload['next_calculation_date']}</p>
  </div>
  <div class='grid'>
    <div class='card'><h3>Signal actuel</h3><div class='big {'signal-on' if signal=='RISK ON' else 'signal-off'}'>{signal}</div><div>{payload['decision']}</div></div>
    <div class='card'><h3>Composite</h3><div class='big'>{payload['pillars']['composite']}</div><div>Score global 0-100</div></div>
    <div class='card'><h3>Credit / Vol</h3><div class='big'>{payload['pillars']['credit_vol']}</div><div>Bloc stress marché</div></div>
    <div class='card'><h3>Tendance</h3><div class='big'>{payload['pillars']['trend'].upper()}</div><div>Confirmation prix mensuelle</div></div>
  </div>
  <div class='pillars'>
    <div class='pill' style='background:{pill_color(payload['pillars']['growth'])}'><small>Growth</small><div class='big'>{payload['pillars']['growth']}</div></div>
    <div class='pill' style='background:{pill_color(payload['pillars']['inflation'])}'><small>Inflation</small><div class='big'>{payload['pillars']['inflation']}</div></div>
    <div class='pill' style='background:{pill_color(payload['pillars']['rates'])}'><small>Rates</small><div class='big'>{payload['pillars']['rates']}</div></div>
    <div class='pill' style='background:{pill_color(payload['pillars']['credit_vol'])}'><small>Credit / Vol</small><div class='big'>{payload['pillars']['credit_vol']}</div></div>
    <div class='pill' style='background:{pill_color(payload['pillars']['composite'])}'><small>Composite</small><div class='big'>{payload['pillars']['composite']}</div></div>
  </div>
  <div class='rules'>
    <h2>Règles du modèle</h2>
    <ul>
      <li>Risk-On si Composite ≥ 48 et Credit/Vol ≥ 45</li>
      <li>Risk-On si tendance positive et Composite ≥ 45 et Credit/Vol ≥ 35</li>
      <li>Risk-Off si Composite ≤ 40 et tendance négative</li>
      <li>Risk-Off si Credit/Vol ≤ 35 et tendance négative</li>
      <li>Sinon : conservation de l’état précédent</li>
    </ul>
  </div>
  <div class='tablebox'>
    <h2>Indicateurs</h2>
    <table>
      <thead><tr><th>Indicateur</th><th>Valeur brute</th><th>Score</th><th>Pourquoi il compte</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
  <div class='footer'>
    Calcul automatique prévu chaque dernier jour du mois après la clôture US. Décision à prendre le lendemain à l’ouverture.<br/>
    Site statique généré par Python et publié via GitHub Pages.
  </div>
</div>
</body>
</html>"""

def main():
    force = os.getenv('FORCE_RUN', '0') == '1'
    now_ny = datetime.now(NY)
    if not force and not is_last_day_of_month(now_ny.date()):
        print("Aujourd'hui n'est pas le dernier jour du mois à New York. Aucun fichier mis à jour.")
        return

    payload = build_payload(now_ny.date())
    with open(os.path.join(DOCS_DIR, 'signal_snapshot.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DOCS_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(render_dashboard(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
