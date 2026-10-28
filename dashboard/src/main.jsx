import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, BarChart3, Bitcoin, BrainCircuit, ChartNoAxesCombined, CircleHelp, Clock3, Database, FileChartColumn, Gauge, List, Radio, RefreshCw, ShieldAlert, SlidersHorizontal, Trophy } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CandlestickSeries, ColorType, createChart, CrosshairMode } from 'lightweight-charts';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const pct = (x, digits = 1) => x !== null && x !== undefined && x !== '' && Number.isFinite(Number(x)) ? `${(Number(x) * 100).toFixed(digits)}%` : '—';
const number = (x, digits = 2) => x !== null && x !== undefined && x !== '' && Number.isFinite(Number(x)) ? Number(x).toFixed(digits) : '—';
const formatPrice = (val, digits = 1) => {
  if (val == null || !Number.isFinite(Number(val))) return '—';
  return `$${Number(val).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
};
const date = (x, opts = {}) => {
  if (!x) return '—';
  const zone = opts.timeZone || 'UTC';
  try {
    const hasExplicit = opts.hour || opts.minute || opts.second || opts.day || opts.month || opts.year;
    const defaultFormat = hasExplicit ? {} : { dateStyle: 'medium', timeStyle: 'short' };
    return new Intl.DateTimeFormat(undefined, { timeZone: zone, ...defaultFormat, ...opts }).format(new Date(x)) + (zone === 'Asia/Karachi' ? ' PKT' : ' UTC');
  } catch (e) {
    return String(x).slice(0, 19).replace('T', ' ');
  }
};

function App() {
  const [data, setData] = useState(null);
  const [candleData, setCandleData] = useState(null);
  const [candleError, setCandleError] = useState('');
  const [error, setError] = useState('');
  const [now, setNow] = useState(new Date());
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefreshAt, setLastRefreshAt] = useState(Date.now());
  const refresh = async () => {
    setRefreshing(true);
    try { const r = await fetch(`${API}/api/dashboard`, { cache: 'no-store' }); if (!r.ok) throw new Error(`API returned ${r.status}`); setData(await r.json()); setLastRefreshAt(Date.now()); setError(''); }
    catch (e) { setError(`${e.message}. Start the API, then refresh.`); }
    finally { setRefreshing(false); }
  };
  const refreshCandles = async () => {
    try { const r = await fetch(`${API}/api/market/candles?limit=168`, { cache: 'no-store' }); if (!r.ok) throw new Error(`Candle API returned ${r.status}`); const result = await r.json(); setCandleData(result); setCandleError(''); }
    catch (e) { setCandleError(e.message); }
  };
  useEffect(() => { refresh(); refreshCandles(); const a = setInterval(refresh, 15_000); const c = setInterval(refreshCandles, 15_000); const b = setInterval(() => setNow(new Date()), 1000); return () => { clearInterval(a); clearInterval(b); clearInterval(c); }; }, []);
  const p = data?.prediction;
  const metrics = data?.final_test || {};
  const scenarios = data?.strategy_scenarios || [];
  const bestScenario = scenarios.find(x => Number(x.threshold) === 0.65);
  const chartData = scenarios.map(x => ({ threshold: `${Math.round(Number(x.threshold) * 100)}%`, result: Number(x.total_return_pct) }));
  const nextRefreshSeconds = Math.max(0, Math.ceil((15_000 - (now.getTime() - lastRefreshAt)) / 1000));
  return <div className="app-shell">
    <main className="main">
      <header className="topbar"><div><div className="eyebrow">BITCOIN · MACHINE LEARNING · RESEARCH</div><h1>Intelligence Dashboard</h1></div><div className="top-actions"><div className="system-status"><span className={`online-dot ${error ? "offline" : refreshing ? "busy" : ""}`}/>{error ? "API DISCONNECTED" : data ? "MARKET LIVE" : "CONNECTING"}<small className="refresh-countdown">{refreshing ? "SYNCING…" : `refresh ${nextRefreshSeconds}s`}</small></div><div className="live-clock"><Clock3 size={15}/>{new Intl.DateTimeFormat(undefined,{timeZone:'UTC',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(now)} <small>UTC · PKT {new Intl.DateTimeFormat(undefined,{timeZone:'Asia/Karachi',hour:'2-digit',minute:'2-digit',hour12:false}).format(now)}</small></div><button className="icon-button" onClick={refresh} title="Refresh now"><RefreshCw size={16} className={refreshing?'spin':''}/></button></div></header>
      {error && <div className="error-banner"><ShieldAlert size={17}/>{error}</div>}
      <section className="metric-grid">
        <Metric icon={<Bitcoin/>} label="BTC / USDT PERPETUAL" sub={p?.price_source === "live_ticker" ? "Live Binance USD-M ticker" : "Latest completed 1h close"} value={p?.current_price ? `$${Number(p.current_price).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}` : 'Waiting for API'} foot={p?.price_change_24h_pct != null ? `${Number(p.price_change_24h_pct)>=0?'+':''}${number(p.price_change_24h_pct)}% · 24h` : 'Live public market feed'} accent="cyan"/>
        <Metric icon={<BrainCircuit/>} label="NEXT CANDLE ESTIMATE" sub={p?.status === 'available' ? `Prediction: ${date(p.predicted_candle_open_utc,{dateStyle:'short',timeStyle:'short'})}` : 'Signal appears when data is valid'} value={p?.direction || '—'} foot={p?.confidence_level ? `${p.confidence_level} · UP ${pct(p.probability_up)}` : (p?.probability_up != null ? `UP ${pct(p.probability_up)}  ·  DOWN ${pct(p.probability_down)}` : p?.status || 'Waiting for API')} accent={p?.direction === 'UP' ? 'green' : p?.direction === 'DOWN' ? 'red' : 'muted'}/>
        <Metric icon={<Gauge/>} label="FINAL TEST · DIRECTION" sub="Ensemble historical holdout" value={metrics.balanced_accuracy != null ? pct(metrics.balanced_accuracy) : '—'} foot={`Accuracy ${pct(metrics.accuracy)} · AUC ${number(metrics.roc_auc,3)}`} accent="blue"/>
        <Metric icon={<Trophy/>} label="STRATEGY RESEARCH" sub="Final test · $10,000 simulation" value={bestScenario ? `${Number(bestScenario.total_return_pct)>=0?'+':''}${number(bestScenario.total_return_pct)}%` : '—'} foot={bestScenario ? `65% threshold · ${bestScenario.long_hours} long / ${bestScenario.short_hours} short hours` : 'See scenarios below'} accent={Number(bestScenario?.total_return_pct)>=0?'green':'red'}/>
      </section>
      <section className="content-grid primary-grid">
        <div className="panel chart-slot"><div className="panel-heading"><div><span className="section-index">01 / MARKET VIEW</span><h2><Radio size={18}/> BTC/USDT Live Chart <span className="chip">1H</span></h2></div><span className={`freshness ${candleData ? 'fresh' : 'stale'}`}><i/>{candleData ? `${candleData.candles?.length || 0} CANDLES` : 'CONNECTING'}</span></div>{candleError && <div className="chart-error">{candleError}. Check that the backend API is running.</div>}<MarketChart candles={candleData?.candles || []}/><div className="chart-foot"><span>Binance USD-M Futures Â· Hover a candle for OHLCV</span><span>Chart refreshed {candleData?.as_of_utc ? date(candleData.as_of_utc,{timeStyle:'short'}) : 'â€”'} Â· {now.toISOString().slice(0,19).replace('T',' ')} UTC</span></div></div>
        <div className="panel prediction-panel"><div className="panel-heading"><div><span className="section-index">02 / MODEL OUTPUT</span><h2><BrainCircuit size={18}/> Latest prediction <HoverInfo text="Features from the last completed candle feed the saved model. Its direction applies to the following one-hour candle."/></h2></div><span className={`freshness ${p?.freshness==='fresh'?'fresh':'stale'}`}><i/>{p?.freshness || 'waiting'}</span></div><div className="prediction-time"><span>MODEL INPUT · LAST CLOSED CANDLE</span><strong>{date(p?.last_closed_candle_open_utc)}</strong><small>UTC: {p?.last_closed_candle_open_utc ? new Date(p.last_closed_candle_open_utc).toISOString().slice(11,16) : '—'}–{p?.last_closed_candle_close_utc ? new Date(p.last_closed_candle_close_utc).toISOString().slice(11,16) : '—'} · PKT: {p?.last_closed_candle_open_utc ? new Date(new Date(p.last_closed_candle_open_utc).getTime()+5*60*60*1000).toISOString().slice(11,16) : '—'}–{p?.last_closed_candle_close_utc ? new Date(new Date(p.last_closed_candle_close_utc).getTime()+5*60*60*1000).toISOString().slice(11,16) : '—'}</small></div><div className={`direction ${p?.direction==='UP'?'up':p?.direction==='DOWN'?'down':''}`}>{p?.direction || (p?.status==='available'?'—':'WAIT')}</div><div className="predicts-label">ESTIMATE FOR THE FOLLOWING CANDLE</div><div className="predicts-range">{p?.predicted_candle_open_utc ? `${date(p.predicted_candle_open_utc)} — ${new Date(p.predicted_candle_close_utc).toISOString().slice(11,16)} UTC` : 'Waiting for valid market data'}</div><div className="predicts-local">PKT: {p?.predicted_candle_open_utc ? `${date(p.predicted_candle_open_utc,{timeZone:'Asia/Karachi'})} — ${date(p.predicted_candle_close_utc,{timeZone:'Asia/Karachi'})}` : '—'}</div><ProbabilityCandle prediction={p}/><div className="time-clarity"><Clock3 size={16}/><span><b>Live clock:</b> {new Intl.DateTimeFormat(undefined,{timeZone:'UTC',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(now)} UTC · {new Intl.DateTimeFormat(undefined,{timeZone:'Asia/Karachi',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(now)} PKT<br/><b>Data age:</b> {p?.data_age_minutes != null ? `${p.data_age_minutes} min` : '—'} · <b>Updated:</b> {p?.data_as_of_utc ? date(p.data_as_of_utc) : '—'}</span></div><div className="disclaimer">{p?.confidence_level ? <span><b>{p.confidence_level}</b> · </span> : ''}Ensemble of LightGBM, XGBoost, CatBoost, and Random Forest. {p?.warning || 'Prediction is an estimate, not a guarantee. No automatic trading is enabled.'}</div></div>
        <PredictedPricePanel prediction={p} metrics={data?.price_model_test_metrics}/>
      </section>
      <section className="content-grid secondary-grid">
        <div className="panel"><div className="panel-heading"><div><span className="section-index">03 / RISK & COSTS</span><h2><BarChart3 size={18}/> Strategy scenarios <HoverInfo text="Historical final-test simulations after the recorded fee and slippage assumptions. This is not a live strategy."/></h2></div><span className="subtle">Final holdout · base cost case</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartData} margin={{top:12,right:10,left:-12,bottom:0}}><defs><linearGradient id="shade" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#31dfc0" stopOpacity={0.28}/><stop offset="95%" stopColor="#31dfc0" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#1c3142" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="threshold" tick={{fill:'#8198ac',fontSize:11}} axisLine={false} tickLine={false}/><YAxis tick={{fill:'#8198ac',fontSize:11}} axisLine={false} tickLine={false} tickFormatter={v=>`${v}%`}/><Tooltip contentStyle={{background:'#0b1925',border:'1px solid #254153',borderRadius:10}} formatter={v=>[`${Number(v).toFixed(2)}%`,'Return']}/><Area type="monotone" dataKey="result" stroke="#31dfc0" strokeWidth={2.5} fill="url(#shade)"/></AreaChart></ResponsiveContainer></div><div className="scenario-list">{scenarios.map(s=><div className="scenario-row" key={s.threshold}><b>{Math.round(Number(s.threshold)*100)}% threshold</b><span>{number(s.long_hours,0)}L / {number(s.short_hours,0)}S hours</span><strong className={Number(s.total_return_pct)>=0?'positive':'negative'}>{Number(s.total_return_pct)>=0?'+':''}{number(s.total_return_pct)}%</strong></div>)}</div><p className="panel-note">A high threshold left very few trades in this holdout. The backtest is historical research, not a profit forecast.</p></div>
        <div className="panel"><div className="panel-heading"><div><span className="section-index">04 / OUT-OF-SAMPLE</span><h2><Activity size={18}/> Model quality <HoverInfo text="Scores come from the one-time chronological holdout. Values close to 50 percent direction accuracy indicate weak separation."/></h2></div><span className="subtle">Final test</span></div><div className="quality-metrics"><Quality label="Balanced accuracy" value={pct(metrics.balanced_accuracy)}/><Quality label="Accuracy" value={pct(metrics.accuracy)}/><Quality label="Precision" value={pct(metrics.precision)}/><Quality label="Recall" value={pct(metrics.recall)}/><Quality label="Brier score · lower is better" value={number(metrics.brier_score,3)}/><Quality label="Log loss · lower is better" value={number(metrics.log_loss,3)}/></div><div className="warning-box"><ShieldAlert size={17}/><span><b>Research finding:</b> this held-out direction score is close to chance. The recorded long/short scenarios show that most active thresholds lost money after the assumed costs; do not read the signal as a trade recommendation.</span></div><p className="panel-note">The one-time holdout was evaluated before the saved model was fitted on all labeled rows. Keep these metrics as historical context; do not tune from that period.</p><div className="model-meta"><span><Database size={15}/> {data?.system?.training_rows?.toLocaleString() || '—'} training examples</span><span><Clock3 size={15}/> Trained {data?.system?.trained_at_utc ? date(data.system.trained_at_utc,{dateStyle:'medium'}) : '—'}</span><span><BrainCircuit size={15}/> {data?.system?.model_version || '—'}</span></div></div>
      </section>
      <section className="content-grid bottom-grid">
        <PredictionLogPanel predictions={data?.recent_predictions} />
        <div className="panel monitor-panel"><div className="panel-heading"><div><span className="section-index">06 / MODEL MONITOR</span><h2><Activity size={18}/> Drift & monitoring</h2></div><span className="freshness fresh"><i/>TRACKING</span></div><div className="monitor-summary"><div className="monitor-number">{data?.monitoring?.months_reported ?? '—'}</div><div><b>historical validation months</b><small>Out-of-fold predictions in monitoring file</small></div></div>{data?.monitoring?.latest&&<div className="monitor-latest"><span>Latest monitored month</span><b>{data.monitoring.latest.month || '—'}</b><span>Accuracy</span><b>{pct(data.monitoring.latest.accuracy)}</b><span>Balanced accuracy</span><b>{pct(data.monitoring.latest.balanced_accuracy)}</b><span>Observations</span><b>{data.monitoring.latest.rows ?? '—'}</b></div>}<p className="panel-note">These are historical validation observations, not live model accuracy. A new live prediction is marked correct/incorrect only after its exact target candle closes.</p></div>
      </section>
      <footer className="footer"><div><span className="online-dot"/> DATA FEED <b>{p?.freshness?.toUpperCase() || 'WAITING'}</b>{p?.data_as_of_utc && <> · AS OF {date(p.data_as_of_utc)}</>}</div><div>MODEL <b>{data?.system?.model_version || '—'}</b></div><div>LIVE CLOCK <b>{now.toISOString().slice(11,19)} UTC</b></div></footer>
    </main>
  </div>;
}

function PredictionLogPanel({ predictions = [] }) {
  return (
    <div className="panel recent-panel">
      <div className="panel-heading">
        <div>
          <span className="section-index">05 / CANDLE VERIFICATION</span>
          <h2>
            <List size={18} /> High & Low: Prediction vs Market
          </h2>
        </div>
        <span className="subtle">Hourly · What was predicted vs What actually happened</span>
      </div>

      <div className="table-scroll simplified-log-scroll">
        <table className="simplified-log-table">
          <thead>
            <tr>
              <th>Candle Time</th>
              <th>AI Signal</th>
              <th>Predicted High</th>
              <th>Actual High</th>
              <th>Predicted Low</th>
              <th>Actual Low</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {(predictions || []).map((row) => {
              const isPending = row.actual_up == null;
              const actualDir = isPending ? null : (Number(row.actual_up) === 1 ? 'UP' : 'DOWN');
              const isHit = actualDir ? actualDir === row.direction : null;
              const winProb = row.direction === 'UP' ? Number(row.probability_up) : 1 - Number(row.probability_up);

              const predHigh = row.predicted_high != null ? Number(row.predicted_high) : null;
              const predLow = row.predicted_low != null ? Number(row.predicted_low) : null;
              const actHigh = row.actual_high != null ? Number(row.actual_high) : null;
              const actLow = row.actual_low != null ? Number(row.actual_low) : null;

              const highDiff = predHigh != null && actHigh != null && !isPending ? actHigh - predHigh : null;
              const lowDiff = predLow != null && actLow != null && !isPending ? actLow - predLow : null;

              return (
                <tr key={row.prediction_candle_open_utc} className={`simple-row ${isPending ? 'row-pending' : isHit ? 'row-correct' : 'row-wrong'}`}>
                  {/* 1. Time */}
                  <td className="cell-time">
                    <div className="time-primary">
                      {date(row.prediction_candle_open_utc, { hour: '2-digit', minute: '2-digit', hour12: false })} – {new Date(row.prediction_candle_close_utc).toISOString().slice(11, 16)} UTC
                    </div>
                    <div className="time-secondary">
                      PKT: {date(row.prediction_candle_open_utc, { timeZone: 'Asia/Karachi', hour: '2-digit', minute: '2-digit', hour12: false })} – {date(row.prediction_candle_close_utc, { timeZone: 'Asia/Karachi', hour: '2-digit', minute: '2-digit', hour12: false })}
                    </div>
                  </td>

                  {/* 2. Signal */}
                  <td className="cell-signal">
                    <div className={`signal-badge ${row.direction === 'UP' ? 'signal-up' : 'signal-down'}`}>
                      {row.direction === 'UP' ? '▲ UP' : '▼ DOWN'}
                    </div>
                    <div className="signal-prob">{(winProb * 100).toFixed(1)}% prob</div>
                  </td>

                  {/* 3. Predicted High */}
                  <td className="cell-price hl-pred-high">
                    <b>{formatPrice(predHigh)}</b>
                  </td>

                  {/* 4. Actual High */}
                  <td className="cell-price hl-actual-high">
                    {isPending ? (
                      <span className="live-chip">LIVE: {formatPrice(actHigh)}</span>
                    ) : (
                      <div>
                        <b>{formatPrice(actHigh)}</b>
                        {highDiff != null && (
                          <span className="diff-pill">
                            Farq: {highDiff >= 0 ? '+' : ''}${highDiff.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )}
                  </td>

                  {/* 5. Predicted Low */}
                  <td className="cell-price hl-pred-low">
                    <b>{formatPrice(predLow)}</b>
                  </td>

                  {/* 6. Actual Low */}
                  <td className="cell-price hl-actual-low">
                    {isPending ? (
                      <span className="live-chip">LIVE: {formatPrice(actLow)}</span>
                    ) : (
                      <div>
                        <b>{formatPrice(actLow)}</b>
                        {lowDiff != null && (
                          <span className="diff-pill">
                            Farq: {lowDiff >= 0 ? '+' : ''}${lowDiff.toFixed(1)}
                          </span>
                        )}
                      </div>
                    )}
                  </td>

                  {/* 7. Result */}
                  <td className="cell-verdict">
                    {isPending ? (
                      <span className="verdict-badge pending-badge">⏳ FORMING</span>
                    ) : isHit ? (
                      <span className="verdict-badge correct-badge">✔ CORRECT</span>
                    ) : (
                      <span className="verdict-badge wrong-badge">✘ WRONG</span>
                    )}
                  </td>
                </tr>
              );
            })}
            {!predictions?.length && (
              <tr>
                <td colSpan="7" className="empty-table">
                  Abhi koi records nahi hain. Agli candle shuru hote hi live data aayega.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ProbabilityCandle({prediction}) {
  const up = prediction?.probability_up == null ? null : Number(prediction.probability_up);
  const down = prediction?.probability_down == null ? null : Number(prediction.probability_down);
  const direction = prediction?.direction;
  const mainProbability = direction === "DOWN" ? down : up;
  const bodyHeight = mainProbability == null ? 50 : Math.max(44, Math.min(76, 44 + Math.abs(mainProbability * 100 - 50) * 1.3));
  return <div className="probability-viz" title="Candle color shows the model's directional class. The two values are raw class probabilities, not predicted percent price movement.">
    <div className={`prediction-candle ${direction === "DOWN" ? "down" : direction === "UP" ? "up" : "idle"}`} style={{height: bodyHeight}} aria-label={`Predicted direction ${direction || "unavailable"}; probability up ${pct(up)}, down ${pct(down)}`}>
      <span className="candle-wick"/><span className="candle-body"><b>{direction || "WAIT"}</b><small>{pct(mainProbability)}</small></span>
    </div>
    <div className="probability-breakdown"><div className="probability-line"><span className="up-label">▲ UP chance</span><b>{pct(up)}</b></div><div className="probability-line"><span className="down-label">▼ DOWN chance</span><b>{pct(down)}</b></div><div className="probability-track"><span className="track-up" style={{width: up == null ? "0%" : pct(up)}}/><span className="track-down" style={{width: down == null ? "0%" : pct(down)}}/></div><small>Ensemble probability · not price-change size</small></div>
    {prediction?.ensemble_probabilities && (
      <div style={{display: 'flex', gap: '6px', justifyContent: 'center', marginTop: '8px', flexWrap: 'wrap'}}>
        {Object.entries(prediction.ensemble_probabilities).map(([name, prob]) => (
          <span key={name} style={{fontSize: '11px', padding: '2px 6px', background: '#132839', borderRadius: '4px', border: '1px solid #1c3d56', color: prob >= 0.5 ? '#31dfc0' : '#ef5967'}}>
            {name === 'random_forest' ? 'RF' : name === 'lightgbm' ? 'LGBM' : name === 'xgboost' ? 'XGB' : 'CAT'}: {(prob * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    )}
  </div>;
}

function PredictedPricePanel({prediction, metrics}) {
  const available = prediction?.status === 'available' && prediction?.estimated_close_price != null;
  const open = Number(prediction?.estimated_open_price);
  const close = Number(prediction?.estimated_close_price);
  const current = Number(prediction?.current_price);
  const high = Number(prediction?.estimated_high_price);
  const low = Number(prediction?.estimated_low_price);
  const direction = prediction?.direction || (close < open ? 'DOWN' : 'UP');
  const isDown = direction === 'DOWN';

  const openTime = prediction?.predicted_candle_open_utc ? new Date(prediction.predicted_candle_open_utc) : null;
  const closeTime = prediction?.predicted_candle_close_utc ? new Date(prediction.predicted_candle_close_utc) : null;
  const remainingMins = closeTime ? Math.max(0, Math.round((closeTime.getTime() - Date.now()) / 60000)) : null;

  const dollarChange = close - open;
  const pctChange = open > 0 ? (dollarChange / open) * 100 : 0;

  return <div className="panel estimated-candle-panel">
    <div className="panel-heading">
      <div>
        <span className="section-index">03 / PRICE FORECAST</span>
        <h2><ChartNoAxesCombined size={18}/> Next Candle Forecast <HoverInfo text="Direction-aligned forecast estimating the next 1h candle Open, High, Low, and Close prices anchored by market volatility (ATR)."/></h2>
      </div>
      <span className={`freshness ${available ? (isDown ? 'down-fresh' : 'fresh') : 'stale'}`}>
        <i/>{available ? `${direction} FORECAST` : 'WAITING'}
      </span>
    </div>

    {/* Timing Bar */}
    <div className="forecast-time-card">
      <div className="forecast-time-row">
        <div className="time-endpoint">
          <span className="endpoint-label">CANDLE OPEN</span>
          <strong>{openTime ? openTime.toISOString().slice(11, 16) : '—'} UTC</strong>
          <small>{openTime ? new Intl.DateTimeFormat(undefined, {timeZone: 'Asia/Karachi', hour: '2-digit', minute: '2-digit', hour12: false}).format(openTime) + ' PKT' : '—'}</small>
        </div>
        <div className="time-duration-pill">
          <Clock3 size={13}/>
          <span>{remainingMins != null && remainingMins <= 60 ? `${remainingMins}m left` : '1h candle'}</span>
        </div>
        <div className="time-endpoint right-align">
          <span className="endpoint-label">TARGET CLOSE</span>
          <strong>{closeTime ? closeTime.toISOString().slice(11, 16) : '—'} UTC</strong>
          <small>{closeTime ? new Intl.DateTimeFormat(undefined, {timeZone: 'Asia/Karachi', hour: '2-digit', minute: '2-digit', hour12: false}).format(closeTime) + ' PKT' : '—'}</small>
        </div>
      </div>
    </div>

    {/* Visual Candlestick Stage */}
    {available ? (
      <div className={`perfect-candle-stage ${isDown ? 'down' : 'up'}`}>
        <div className="candle-extremes-col">
          <div className="extreme-tag high">
            <span>EST. HIGH</span>
            <b>${high.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1})}</b>
          </div>
          <div className="extreme-tag low">
            <span>EST. LOW</span>
            <b>${low.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1})}</b>
          </div>
        </div>

        <div className="perfect-candle-wrapper">
          <div className="wick upper-wick"></div>
          <div className={`candle-core ${isDown ? 'down' : 'up'}`}>
            <div className="core-badge">
              <b>{isDown ? '▼ BEARISH' : '▲ BULLISH'}</b>
              <span>{Math.abs(pctChange).toFixed(2)}%</span>
            </div>
          </div>
          <div className="wick lower-wick"></div>
        </div>

        <div className="candle-targets-col">
          <div className={`price-target-card ${isDown ? 'open-card' : 'target-card'}`}>
            <span className="card-lbl">OPEN PRICE</span>
            <strong className="price-val">${open.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</strong>
          </div>
          <div className={`price-target-card ${isDown ? 'target-card' : 'open-card'}`}>
            <span className="card-lbl">TARGET CLOSE</span>
            <strong className={`price-val highlight ${isDown ? 'negative' : 'positive'}`}>
              ${close.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
            </strong>
          </div>
        </div>
      </div>
    ) : (
      <div className="estimate-wait">Price scenario will appear when the live prediction is available.</div>
    )}

    {/* Delta Bar */}
    {available && (
      <div className="forecast-delta-bar">
        <div>
          <span>EXPECTED MOVE</span>
          <b className={isDown ? 'negative' : 'positive'}>
            {dollarChange >= 0 ? '+' : ''}${number(dollarChange, 2)} ({dollarChange >= 0 ? '+' : ''}{number(pctChange, 2)}%)
          </b>
        </div>
        <div>
          <span>CURRENT TICKER</span>
          <b className="ticker-val">${current ? current.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '—'}</b>
        </div>
      </div>
    )}

    {metrics?.available && (
      <div className="price-model-test">
        <b>Model Benchmark</b>
        <span>Holdout MAE: {number(metrics.selected_close_mae_bps, 2)} bps (~${number(metrics.selected_close_mae_dollars_at_80000, 2)} at $80k)</span>
        <span>Baseline MAE: {number(metrics.baseline_close_mae_bps, 2)} bps</span>
      </div>
    )}

    <div className="estimate-caveat">
      Calculated from multi-model ensemble direction combined with Binance USD-M volatility (ATR).
    </div>
  </div>;
}

function MarketChart({candles}) {
  const containerRef = React.useRef(null);
  const candlesRef = React.useRef(candles);
  const chartRef = React.useRef(null);
  const seriesRef = React.useRef(null);
  const initializedRef = React.useRef(false);
  const [hoveredCandle, setHoveredCandle] = useState(null);
  candlesRef.current = candles;
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: '#091621' }, textColor: '#8ca4b5', fontFamily: 'DM Mono, monospace' },
      grid: { vertLines: { color: '#142836' }, horzLines: { color: '#142836' } },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: '#48cbbb', labelBackgroundColor: '#12665f' }, horzLine: { color: '#48cbbb', labelBackgroundColor: '#12665f' } },
      rightPriceScale: { borderColor: '#203746' }, timeScale: { borderColor: '#203746', timeVisible: true, secondsVisible: false, rightOffset: 3 },
      localization: { timeFormatter: time => new Date(Number(time) * 1000).toLocaleString(undefined, { timeZone: 'UTC', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }) + ' UTC' },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: '#21c99a', downColor: '#ef5967', borderUpColor: '#21c99a', borderDownColor: '#ef5967', wickUpColor: '#39dfb1', wickDownColor: '#ff7881',
      priceLineVisible: true, lastValueVisible: true,
    });
    chart.subscribeCrosshairMove(param => {
      if (!param.time || !param.seriesData) { setHoveredCandle(null); return; }
      const candle = param.seriesData.get(series);
      if (!candle) { setHoveredCandle(null); return; }
      const timestamp = typeof param.time === 'number' ? param.time : null;
      const source = candlesRef.current.find(item => item.time === timestamp);
      setHoveredCandle(source ? {...source, ...candle} : null);
    });
    chartRef.current = chart; seriesRef.current = series; initializedRef.current = false;
    const resize = new ResizeObserver(() => { if (!initializedRef.current) chart.timeScale().fitContent(); });
    resize.observe(containerRef.current);
    return () => { resize.disconnect(); chart.remove(); chartRef.current = null; seriesRef.current = null; };
  }, []);
  useEffect(() => {
    if (!seriesRef.current || !candles.length) return;
    const normalized = candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close }));
    if (!initializedRef.current) {
      seriesRef.current.setData(normalized);
      chartRef.current?.timeScale().fitContent();
      initializedRef.current = true;
    } else {
      // Updating only the newest bar keeps the user's zoom and crosshair stable.
      seriesRef.current.update(normalized[normalized.length - 1]);
    }
  }, [candles]);
  const latest = candles.at(-1);
  const displayCandle = hoveredCandle || latest;
  return <div className="market-chart-area">
    <div ref={containerRef} className="market-chart" role="img" aria-label="BTC USDT one hour candlestick chart" />
    {displayCandle && <div className={`forming-candle ${displayCandle.is_closed ? 'closed' : ''}`}><i/>{hoveredCandle ? 'HOVERED CANDLE' : latest.is_closed ? 'LAST CANDLE CLOSED' : 'LIVE CANDLE FORMING'} <b>{date(displayCandle.timestamp_utc,{dateStyle:'short',timeStyle:'short'})}</b> Â· O {number(displayCandle.open,1)} H {number(displayCandle.high,1)} L {number(displayCandle.low,1)} C {number(displayCandle.close,1)} Â· Vol {number(displayCandle.volume,2)}</div>}
    {!candles.length && <div className="chart-loading"><RefreshCw size={17} className="spin"/> Waiting for candle dataâ€¦</div>}
  </div>;
}

function HoverInfo({text}) { return <span className="hover-info" tabIndex="0" aria-label={text} data-tip={text}><CircleHelp size={14}/></span>; }

function Metric({icon,label,sub,value,foot,accent}) {
  const help = {
    "BTC / USDT PERPETUAL": "Live public Futures ticker price. The ML input still uses the last fully closed candle.",
    "NEXT CANDLE ESTIMATE": "Model direction for the next one-hour candle. The estimate changes when a new candle closes.",
    "FINAL TEST · DIRECTION": "Balanced accuracy averages UP and DOWN class accuracy on the one-time held-out period.",
    "STRATEGY RESEARCH": "A historical simulation at the 65% threshold. The holdout produced only two long position-hours.",
  };
  return <div className="metric-card hover-lift" tabIndex="0"><div className="metric-top"><span className="metric-icon">{icon}</span><span className="metric-label">{label}</span><HoverInfo text={help[label] || "Research metric from this project."}/></div><div className={`metric-value ${accent}`}>{value}</div><div className="metric-sub">{sub}</div><div className="metric-foot">{foot}</div></div>;
}
function Quality({label,value}) { const help = label.includes("Balanced") ? "Mean of UP recall and DOWN recall; 50% is chance-level for a balanced binary target." : label.includes("Brier") || label.includes("Log loss") ? "Probability scoring rules; smaller is better. These raw model probabilities are not calibrated." : "Classification metric on the chronological final test."; return <div className="quality-item hover-explain" tabIndex="0" data-tip={help}><span>{label}</span><b>{value}</b></div>; }

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
