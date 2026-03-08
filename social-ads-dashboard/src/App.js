import { useState, useEffect, useRef } from "react";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  ScatterChart, Scatter, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine
} from "recharts";

// ── DATA ──────────────────────────────────────────────────────
const CHANNEL_DATA = [
  { channel: "Instagram", roi: 4.01, revenue: 2.33, profit: 1.75, profitable: 87, engagement: 5.51 },
  { channel: "Facebook",  roi: 3.99, revenue: 2.32, profit: 1.74, profitable: 87, engagement: 5.48 },
  { channel: "Twitter",   roi: 4.01, revenue: 2.32, profit: 1.74, profitable: 88, engagement: 5.50 },
  { channel: "Pinterest", roi: 0.72, revenue: 0.42, profit: -0.17, profitable: 30, engagement: 1.00 },
];

const MONTHLY_DATA = [
  { month: "Jan", revenue: 627, profit: 429, spend: 198 },
  { month: "Feb", revenue: 569, profit: 388, spend: 181 },
  { month: "Mar", revenue: 637, profit: 438, spend: 199 },
  { month: "Apr", revenue: 612, profit: 421, spend: 191 },
  { month: "May", revenue: 620, profit: 424, spend: 196 },
  { month: "Jun", revenue: 613, profit: 420, spend: 193 },
  { month: "Jul", revenue: 629, profit: 431, spend: 198 },
  { month: "Aug", revenue: 619, profit: 422, spend: 197 },
  { month: "Sep", revenue: 608, profit: 417, spend: 191 },
  { month: "Oct", revenue: 625, profit: 428, spend: 197 },
  { month: "Nov", revenue: 601, profit: 412, spend: 189 },
  { month: "Dec", revenue: 627, profit: 430, spend: 197 },
];

const SEGMENT_DATA = [
  { segment: "Health",     revenue: 1.48, roi: 3.19, engagement: 4.38 },
  { segment: "Technology", revenue: 1.48, roi: 3.19, engagement: 4.37 },
  { segment: "Fashion",    revenue: 1.48, roi: 3.18, engagement: 4.38 },
  { segment: "Home",       revenue: 1.48, roi: 3.17, engagement: 4.35 },
  { segment: "Food",       revenue: 1.47, roi: 3.16, engagement: 4.36 },
];

const MODEL_DATA = [
  { model: "Logistic",   highROI: 0.7528, highProfit: 0.8565, success: 0.9739 },
  { model: "Rnd Forest", highROI: 0.7549, highProfit: 0.8567, success: 0.9882 },
  { model: "Grad Boost", highROI: 0.7537, highProfit: 0.8569, success: 0.9888 },
  { model: "XGBoost",    highROI: 0.7547, highProfit: 0.8568, success: 0.9886 },
  { model: "AdaBoost",   highROI: 0.7552, highProfit: 0.8571, success: 0.9885 },
];

const FEATURE_DATA = [
  { feature: "CPC",              importance: 0.241 },
  { feature: "Engagement×CTR",   importance: 0.207 },
  { feature: "CPM",              importance: 0.198 },
  { feature: "Channel Score",    importance: 0.181 },
  { feature: "CTR",              importance: 0.070 },
  { feature: "Channel Enc",      importance: 0.044 },
  { feature: "Engagement Score", importance: 0.014 },
  { feature: "Impressions",      importance: 0.010 },
];

const GENDER_DATA = [
  { name: "Women", revenue: 4.11, profit: 2.82 },
  { name: "Men",   revenue: 3.27, profit: 2.24 },
];

const AGE_DATA = [
  { age: "18-24", roi: 3.18 },
  { age: "25-34", roi: 3.17 },
  { age: "35-44", roi: 3.17 },
  { age: "45-60", roi: 3.18 },
];

const LOCATION_DATA = [
  { location: "Miami",       revenue: 1.49, roi: 3.18 },
  { location: "LA",          revenue: 1.48, roi: 3.18 },
  { location: "New York",    revenue: 1.48, roi: 3.19 },
  { location: "Austin",      revenue: 1.47, roi: 3.17 },
  { location: "Las Vegas",   revenue: 1.47, roi: 3.18 },
];

// ── THEME ─────────────────────────────────────────────────────
const T = {
  bg:       "#0a0e1a",
  surface:  "#111827",
  card:     "#161d2e",
  border:   "#1e2d45",
  accent:   "#3b82f6",
  green:    "#10b981",
  red:      "#ef4444",
  orange:   "#f59e0b",
  purple:   "#8b5cf6",
  teal:     "#06b6d4",
  text:     "#f1f5f9",
  muted:    "#64748b",
  dim:      "#94a3b8",
};

const CHART_COLORS = [T.accent, T.green, T.orange, T.purple, T.teal];

// ── HELPERS ───────────────────────────────────────────────────
const fmt = {
  B: v => `$${v.toFixed(2)}B`,
  M: v => `$${Math.round(v)}M`,
  pct: v => `${v.toFixed(1)}%`,
  num: v => v.toLocaleString(),
};

const CustomTooltip = ({ active, payload, label, prefix = "", suffix = "" }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: T.card, border: `1px solid ${T.border}`,
      borderRadius: 8, padding: "10px 14px", fontSize: 12,
    }}>
      <p style={{ color: T.dim, marginBottom: 6, fontWeight: 600 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, margin: "2px 0" }}>
          {p.name}: <b>{prefix}{typeof p.value === "number" ? p.value.toFixed(3) : p.value}{suffix}</b>
        </p>
      ))}
    </div>
  );
};

// ── PREDICTION ENGINE (rule-based mirror of model logic) ──────
function predict(inputs) {
  const { channel, goal, segment, gender, age, duration,
          acquisition, clicks, impressions, engagement } = inputs;

  const channelScore = { Instagram: 3, Facebook: 2, Twitter: 2, Pinterest: 0 }[channel] ?? 1;
  const ctr = clicks > 0 && impressions > 0 ? (clicks / impressions) * 100 : 0;
  const cpc = acquisition > 0 && clicks > 0 ? acquisition / clicks : 0;
  const cpm = acquisition > 0 && impressions > 0 ? (acquisition / impressions) * 1000 : 0;
  const engCTR = engagement * ctr;

  // High ROI — driven by CPC, CPM, Channel Score, Engagement×CTR
  const roiScore =
    (channelScore / 3) * 0.35 +
    (engCTR > 150 ? 1 : engCTR / 150) * 0.30 +
    (cpc < 0.5 ? 1 : 0.5 / cpc) * 0.20 +
    (cpm < 130 ? 1 : 130 / cpm) * 0.15;
  const highROI = Math.min(0.97, Math.max(0.03, roiScore));

  // High Profit — acquisition cost + engagement
  const profitScore =
    (channelScore / 3) * 0.30 +
    (engagement / 10) * 0.25 +
    (acquisition > 5000 ? Math.min(1, acquisition / 15000) : 0.3) * 0.25 +
    (duration > 30 ? 1 : duration / 30) * 0.20;
  const highProfit = Math.min(0.97, Math.max(0.03, profitScore));

  // Campaign Success — best model AUC 0.9888
  const successScore =
    (channelScore / 3) * 0.40 +
    (engagement / 10) * 0.30 +
    (ctr > 30 ? 1 : ctr / 30) * 0.20 +
    (duration > 30 ? 0.8 : 0.5) * 0.10;
  const success = Math.min(0.97, Math.max(0.03, successScore));

  return {
    highROI:    +(highROI * 100).toFixed(1),
    highProfit: +(highProfit * 100).toFixed(1),
    success:    +(success * 100).toFixed(1),
  };
}

// ── COMPONENTS ────────────────────────────────────────────────
function Card({ children, style = {} }) {
  return (
    <div style={{
      background: T.card, border: `1px solid ${T.border}`,
      borderRadius: 12, padding: 20, ...style,
    }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{
      fontSize: 13, fontWeight: 700, letterSpacing: "0.12em",
      textTransform: "uppercase", color: T.accent,
      marginBottom: 20, display: "flex", alignItems: "center", gap: 8,
    }}>
      <span style={{
        width: 3, height: 16, background: T.accent,
        borderRadius: 2, display: "inline-block",
      }} />
      {children}
    </h2>
  );
}

function KPICard({ label, value, sub, color = T.accent }) {
  return (
    <Card style={{ textAlign: "center" }}>
      <div style={{ fontSize: 28, fontWeight: 800, color, fontFamily: "'Space Mono', monospace" }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: T.muted, marginTop: 4, fontWeight: 600 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: T.dim, marginTop: 2 }}>{sub}</div>}
    </Card>
  );
}

function PredictBar({ label, value, color }) {
  const [width, setWidth] = useState(0);
  useEffect(() => { setTimeout(() => setWidth(value), 100); }, [value]);
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 13, color: T.dim }}>{label}</span>
        <span style={{ fontSize: 14, fontWeight: 700, color, fontFamily: "'Space Mono', monospace" }}>
          {value}%
        </span>
      </div>
      <div style={{ background: T.border, borderRadius: 4, height: 8 }}>
        <div style={{
          width: `${width}%`, height: "100%", borderRadius: 4,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)",
          boxShadow: `0 0 8px ${color}66`,
        }} />
      </div>
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ fontSize: 11, color: T.muted, display: "block", marginBottom: 5, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        {label}
      </label>
      <select value={value} onChange={e => onChange(e.target.value)} style={{
        width: "100%", background: T.surface, border: `1px solid ${T.border}`,
        color: T.text, borderRadius: 7, padding: "8px 10px", fontSize: 13,
        cursor: "pointer", outline: "none",
      }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function NumberInput({ label, value, onChange, min, max, step = 1 }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ fontSize: 11, color: T.muted, display: "block", marginBottom: 5, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        {label}
      </label>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={e => onChange(Number(e.target.value))}
        style={{
          width: "100%", background: T.surface, border: `1px solid ${T.border}`,
          color: T.text, borderRadius: 7, padding: "8px 10px", fontSize: 13,
          outline: "none", boxSizing: "border-box",
        }} />
    </div>
  );
}

// ── SECTIONS ──────────────────────────────────────────────────

function Overview() {
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 20 }}>
        <KPICard label="Total Revenue"    value="$7.39B"  sub="All channels 2022"  color={T.accent} />
        <KPICard label="Total Profit"     value="$5.06B"  sub="68.6% margin"       color={T.green}  />
        <KPICard label="Total Campaigns"  value="254,960" sub="Across 50 companies" color={T.purple} />
        <KPICard label="Avg ROI"          value="3.18"    sub="Excl. Pinterest: 4.0" color={T.orange} />
        <KPICard label="Avg CTR"          value="31.42%"  sub="Pinterest: 29.24%"  color={T.teal}   />
        <KPICard label="Profitable"       value="73.1%"   sub="219K / 300K campaigns" color={T.green} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card>
          <SectionTitle>Monthly Revenue vs Profit ($M)</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={MONTHLY_DATA}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="month" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip suffix="M" />} />
              <Legend wrapperStyle={{ fontSize: 12, color: T.dim }} />
              <Line type="monotone" dataKey="revenue" stroke={T.accent} strokeWidth={2} dot={{ r: 3 }} name="Revenue" />
              <Line type="monotone" dataKey="profit"  stroke={T.green}  strokeWidth={2} dot={{ r: 3 }} name="Profit" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Revenue by Channel ($B)</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={CHANNEL_DATA} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="channel" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip prefix="$" suffix="B" />} />
              <Bar dataKey="revenue" name="Revenue" radius={[4, 4, 0, 0]}>
                {CHANNEL_DATA.map((e, i) => (
                  <Cell key={i} fill={e.channel === "Pinterest" ? T.red : T.accent} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

function PinterestSection() {
  const compData = CHANNEL_DATA.map(c => ({
    ...c, profitColor: c.profit < 0 ? T.red : T.green,
  }));

  return (
    <div>
      <Card style={{ marginBottom: 14, background: "linear-gradient(135deg, #1a0a0a, #1e1020)", borderColor: "#3d1a1a" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontSize: 28 }}>⚠️</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: T.red }}>Pinterest is losing $165M</div>
            <div style={{ fontSize: 13, color: T.dim, marginTop: 3 }}>
              ROI of 0.72 vs 4.0 for every other channel. Only 30% of campaigns are profitable.
              Engagement score is 1.0 — effectively dead. CPC is $1.71 vs $0.39 elsewhere.
            </div>
          </div>
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        <Card>
          <SectionTitle>Total Profit by Channel ($B)</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={CHANNEL_DATA} barSize={30}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="channel" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip prefix="$" suffix="B" />} />
              <ReferenceLine y={0} stroke={T.muted} strokeWidth={1} />
              <Bar dataKey="profit" name="Profit" radius={[4, 4, 0, 0]}>
                {CHANNEL_DATA.map((e, i) => (
                  <Cell key={i} fill={e.profit < 0 ? T.red : T.green} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Avg Engagement Score</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={CHANNEL_DATA} barSize={30}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="channel" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="engagement" name="Engagement" radius={[4, 4, 0, 0]}>
                {CHANNEL_DATA.map((e, i) => (
                  <Cell key={i} fill={e.channel === "Pinterest" ? T.red : T.teal} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { label: "Pinterest ROI",        value: "0.72",  sub: "vs 4.01 avg",  color: T.red    },
          { label: "Profitable Campaigns", value: "30%",   sub: "vs 87% avg",   color: T.red    },
          { label: "Engagement Score",     value: "1.0",   sub: "vs 5.5 avg",   color: T.red    },
          { label: "Total Loss",           value: "$165M", sub: "2022 total",   color: T.orange },
        ].map((k, i) => <KPICard key={i} {...k} />)}
      </div>
    </div>
  );
}

function AudienceSection() {
  const genderPie = [
    { name: "Women", value: 4.11 },
    { name: "Men",   value: 3.27 },
  ];

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 14 }}>
        <Card>
          <SectionTitle>Revenue by Segment ($B)</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={SEGMENT_DATA} layout="vertical" barSize={16}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} horizontal={false} />
              <XAxis type="number" tick={{ fill: T.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="segment" tick={{ fill: T.dim, fontSize: 11 }} axisLine={false} tickLine={false} width={70} />
              <Tooltip content={<CustomTooltip prefix="$" suffix="B" />} />
              <Bar dataKey="revenue" name="Revenue" radius={[0, 4, 4, 0]}>
                {SEGMENT_DATA.map((_, i) => <Cell key={i} fill={CHART_COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Revenue Split by Gender</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={genderPie} cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                dataKey="value" nameKey="name" label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                labelLine={{ stroke: T.muted }} fontSize={11}>
                <Cell fill={T.accent} />
                <Cell fill={T.pink || "#ec4899"} />
              </Pie>
              <Tooltip formatter={v => `$${v}B`} />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>ROI by Age Group</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={AGE_DATA} barSize={30}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="age" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis domain={[3.1, 3.25]} tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="roi" name="Avg ROI" radius={[4, 4, 0, 0]} fill={T.purple} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Card>
          <SectionTitle>Revenue by Location ($B)</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={LOCATION_DATA} barSize={28}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="location" tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: T.muted, fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip prefix="$" suffix="B" />} />
              <Bar dataKey="revenue" name="Revenue" fill={T.teal} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Segment ROI Radar</SectionTitle>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={SEGMENT_DATA}>
              <PolarGrid stroke={T.border} />
              <PolarAngleAxis dataKey="segment" tick={{ fill: T.dim, fontSize: 10 }} />
              <Radar name="ROI" dataKey="roi" stroke={T.accent} fill={T.accent} fillOpacity={0.2} strokeWidth={2} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

function ModelsSection() {
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        <Card>
          <SectionTitle>AUC Score by Model — All Targets</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={MODEL_DATA} barSize={14}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
              <XAxis dataKey="model" tick={{ fill: T.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0.7, 1.02]} tick={{ fill: T.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11, color: T.dim }} />
              <Bar dataKey="highROI"    name="High ROI"    fill={T.orange} radius={[2, 2, 0, 0]} />
              <Bar dataKey="highProfit" name="High Profit" fill={T.accent} radius={[2, 2, 0, 0]} />
              <Bar dataKey="success"    name="Success"     fill={T.green}  radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <SectionTitle>Feature Importance — Random Forest (High ROI)</SectionTitle>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={FEATURE_DATA} layout="vertical" barSize={14}>
              <CartesianGrid strokeDasharray="3 3" stroke={T.border} horizontal={false} />
              <XAxis type="number" tick={{ fill: T.muted, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="feature" tick={{ fill: T.dim, fontSize: 10 }} axisLine={false} tickLine={false} width={100} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="importance" name="Importance" fill={T.purple} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        {[
          { target: "Campaign Success", model: "Gradient Boosting", auc: "0.9888", color: T.green  },
          { target: "High Profit",      model: "AdaBoost",          auc: "0.8571", color: T.accent },
          { target: "High ROI",         model: "AdaBoost",          auc: "0.7552", color: T.orange },
        ].map((m, i) => (
          <Card key={i} style={{ textAlign: "center" }}>
            <div style={{ fontSize: 11, color: T.muted, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Best for {m.target}
            </div>
            <div style={{ fontSize: 22, fontWeight: 800, color: m.color, fontFamily: "'Space Mono', monospace" }}>
              AUC {m.auc}
            </div>
            <div style={{ fontSize: 12, color: T.dim, marginTop: 4 }}>🏆 {m.model}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

function PredictorSection() {
  const [inputs, setInputs] = useState({
    channel: "Instagram", goal: "Brand Awareness",
    segment: "Health", gender: "Women", age: "25-34",
    duration: 30, acquisition: 5000,
    clicks: 15000, impressions: 50000, engagement: 7,
  });
  const [result, setResult] = useState(null);

  const set = (k, v) => setInputs(p => ({ ...p, [k]: v }));

  const handlePredict = () => setResult(predict(inputs));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
      <Card>
        <SectionTitle>Campaign Parameters</SectionTitle>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 16px" }}>
          <Select label="Channel" value={inputs.channel} onChange={v => set("channel", v)}
            options={["Instagram","Facebook","Twitter","Pinterest"].map(v => ({ value: v, label: v }))} />
          <Select label="Campaign Goal" value={inputs.goal} onChange={v => set("goal", v)}
            options={["Brand Awareness","Product Launch","Increase Sales","Market Expansion"].map(v => ({ value: v, label: v }))} />
          <Select label="Customer Segment" value={inputs.segment} onChange={v => set("segment", v)}
            options={["Health","Technology","Fashion","Home","Food"].map(v => ({ value: v, label: v }))} />
          <Select label="Gender" value={inputs.gender} onChange={v => set("gender", v)}
            options={["Women","Men"].map(v => ({ value: v, label: v }))} />
          <Select label="Age Group" value={inputs.age} onChange={v => set("age", v)}
            options={["18-24","25-34","35-44","45-60"].map(v => ({ value: v, label: v }))} />
          <NumberInput label="Duration (Days)" value={inputs.duration} min={15} max={60} step={15} onChange={v => set("duration", v)} />
          <NumberInput label="Acquisition Cost ($)" value={inputs.acquisition} min={500} max={15000} step={500} onChange={v => set("acquisition", v)} />
          <NumberInput label="Engagement Score (1-10)" value={inputs.engagement} min={1} max={10} onChange={v => set("engagement", v)} />
          <NumberInput label="Clicks" value={inputs.clicks} min={300} max={40000} step={1000} onChange={v => set("clicks", v)} />
          <NumberInput label="Impressions" value={inputs.impressions} min={2000} max={120000} step={5000} onChange={v => set("impressions", v)} />
        </div>
        <button onClick={handlePredict} style={{
          width: "100%", marginTop: 8, padding: "12px 0",
          background: `linear-gradient(135deg, ${T.accent}, ${T.purple})`,
          border: "none", borderRadius: 8, color: "white",
          fontSize: 14, fontWeight: 700, cursor: "pointer",
          letterSpacing: "0.06em", textTransform: "uppercase",
        }}>
          ⚡ Predict Campaign Outcome
        </button>
      </Card>

      <Card>
        <SectionTitle>Prediction Results</SectionTitle>
        {!result ? (
          <div style={{
            height: 300, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 12,
          }}>
            <div style={{ fontSize: 40 }}>🎯</div>
            <div style={{ color: T.muted, fontSize: 13, textAlign: "center" }}>
              Fill in campaign parameters and click<br />
              <b style={{ color: T.dim }}>Predict Campaign Outcome</b>
            </div>
          </div>
        ) : (
          <div>
            <PredictBar label="Campaign Success Probability"  value={result.success}    color={T.green}  />
            <PredictBar label="High Profit Probability"       value={result.highProfit} color={T.accent} />
            <PredictBar label="High ROI Probability"          value={result.highROI}    color={T.orange} />

            <div style={{
              marginTop: 20, padding: 14,
              background: T.surface, borderRadius: 8,
              border: `1px solid ${result.success > 65 ? T.green : T.orange}33`,
            }}>
              <div style={{ fontSize: 12, color: T.muted, marginBottom: 8, fontWeight: 600 }}>VERDICT</div>
              {result.success > 65 ? (
                <div style={{ color: T.green, fontSize: 13 }}>
                  ✅ <b>Likely to succeed.</b> High engagement + channel score suggest strong performance.
                </div>
              ) : result.success > 40 ? (
                <div style={{ color: T.orange, fontSize: 13 }}>
                  ⚠️ <b>Moderate probability.</b> Consider switching channel or increasing engagement.
                </div>
              ) : (
                <div style={{ color: T.red, fontSize: 13 }}>
                  ❌ <b>Low probability.</b> Pinterest or low engagement likely dragging performance.
                </div>
              )}
              {inputs.channel === "Pinterest" && (
                <div style={{ color: T.red, fontSize: 12, marginTop: 8 }}>
                  🚨 Pinterest alert: historically $165M in losses. Switch to Instagram or Twitter.
                </div>
              )}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 14 }}>
              {[
                { label: "Est. CTR",  value: `${((inputs.clicks/inputs.impressions)*100).toFixed(1)}%`, color: T.teal },
                { label: "Est. CPC",  value: `$${(inputs.acquisition/inputs.clicks).toFixed(2)}`,       color: T.accent },
                { label: "Est. CPM",  value: `$${(inputs.acquisition/inputs.impressions*1000).toFixed(1)}`, color: T.purple },
                { label: "Duration",  value: `${inputs.duration} days`,                                  color: T.orange },
              ].map((s, i) => (
                <div key={i} style={{
                  background: T.bg, borderRadius: 7, padding: "10px 12px",
                  border: `1px solid ${T.border}`,
                }}>
                  <div style={{ fontSize: 10, color: T.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>{s.label}</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: s.color, marginTop: 3, fontFamily: "'Space Mono', monospace" }}>{s.value}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── MAIN APP ──────────────────────────────────────────────────
const NAV = [
  { id: "overview",   label: "Overview",    icon: "📊" },
  { id: "pinterest",  label: "Pinterest",   icon: "⚠️" },
  { id: "audience",   label: "Audience",    icon: "👥" },
  { id: "models",     label: "ML Models",   icon: "🤖" },
  { id: "predictor",  label: "Predictor",   icon: "⚡" },
];

export default function App() {
  const [active, setActive] = useState("overview");

  const sections = {
    overview:  <Overview />,
    pinterest: <PinterestSection />,
    audience:  <AudienceSection />,
    models:    <ModelsSection />,
    predictor: <PredictorSection />,
  };

  return (
    <div style={{
      minHeight: "100vh", background: T.bg,
      color: T.text, fontFamily: "'DM Sans', sans-serif",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; } 
        ::-webkit-scrollbar-track { background: ${T.bg}; }
        ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 3px; }
        select option { background: ${T.surface}; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.5; }
      `}</style>

      {/* HEADER */}
      <div style={{
        background: T.surface, borderBottom: `1px solid ${T.border}`,
        padding: "0 32px", position: "sticky", top: 0, zIndex: 100,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        height: 58,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: `linear-gradient(135deg, ${T.accent}, ${T.purple})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16,
          }}>📈</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, letterSpacing: "-0.02em" }}>
              Social Media Ads Analytics
            </div>
            <div style={{ fontSize: 10, color: T.muted, letterSpacing: "0.06em" }}>
              300K CAMPAIGNS · 2022 · 50 COMPANIES
            </div>
          </div>
        </div>

        <a href="https://github.com/Ashutosh-AIBOT" target="_blank" rel="noopener noreferrer"
          style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "7px 14px", borderRadius: 8,
            border: `1px solid ${T.border}`, background: T.card,
            color: T.text, textDecoration: "none", fontSize: 13, fontWeight: 600,
            transition: "border-color 0.2s",
          }}
          onMouseEnter={e => e.currentTarget.style.borderColor = T.accent}
          onMouseLeave={e => e.currentTarget.style.borderColor = T.border}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill={T.dim}>
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
          </svg>
          Ashutosh-AIBOT
        </a>
      </div>

      {/* NAV */}
      <div style={{
        background: T.surface, borderBottom: `1px solid ${T.border}`,
        padding: "0 32px", display: "flex", gap: 2,
      }}>
        {NAV.map(n => (
          <button key={n.id} onClick={() => setActive(n.id)} style={{
            padding: "12px 18px", background: "none", border: "none",
            color: active === n.id ? T.accent : T.muted,
            fontSize: 13, fontWeight: active === n.id ? 700 : 500,
            cursor: "pointer", borderBottom: `2px solid ${active === n.id ? T.accent : "transparent"}`,
            transition: "all 0.15s", display: "flex", alignItems: "center", gap: 6,
          }}>
            <span>{n.icon}</span> {n.label}
          </button>
        ))}
      </div>

      {/* CONTENT */}
      <div style={{ padding: "24px 32px", maxWidth: 1200, margin: "0 auto" }}>
        {sections[active]}
      </div>

      {/* FOOTER */}
      <div style={{
        borderTop: `1px solid ${T.border}`, padding: "16px 32px",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginTop: 32,
      }}>
        <div style={{ fontSize: 12, color: T.muted }}>
          Built by <b style={{ color: T.dim }}>Ashutosh</b> · Social Media Ads Analytics Project · 2022 Dataset · 300K Campaigns
        </div>
        <a href="https://github.com/Ashutosh-AIBOT" target="_blank" rel="noopener noreferrer"
          style={{ fontSize: 12, color: T.accent, textDecoration: "none" }}>
          github.com/Ashutosh-AIBOT →
        </a>
        
      </div>
    </div>
  );
}
