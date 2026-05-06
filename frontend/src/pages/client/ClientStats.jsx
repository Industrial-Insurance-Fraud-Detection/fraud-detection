import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

const STATUS_COLORS = {
  APPROVED: '#1A7A4A',
  REJECTED: '#C0392B',
  HUMAN_REVIEW: '#F39C12',
  ANALYZING: '#1A5276',
  PENDING: '#7D6608',
}

const STATUS_LABELS = {
  APPROVED: 'Approuvé',
  REJECTED: 'Rejeté',
  HUMAN_REVIEW: 'Révision',
  ANALYZING: 'Analyse',
  PENDING: 'En attente',
}

function extractClaims(responseData) {
  const inner = responseData?.data ?? responseData
  const arr = inner?.data ?? inner
  return Array.isArray(arr) ? arr : []
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label, dark }) {
  if (!active || !payload || !payload.length) return null
  const bg = dark ? '#111C30' : 'white'
  const border = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div style={{ backgroundColor: bg, border: `1px solid ${border}`, borderRadius: 10, padding: '0.75rem 1rem', boxShadow: '0 8px 24px rgba(0,0,0,0.15)' }}>
      <p style={{ fontSize: '0.78rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ fontSize: '0.75rem', color: p.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: '0.1rem 0' }}>
          {p.name}: <strong>{typeof p.value === 'number' && p.value > 1000 ? `${(p.value / 1000).toFixed(0)}K` : p.value}</strong>
        </p>
      ))}
    </div>
  )
}

// ── Custom Pie label ───────────────────────────────────────────────────────────
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  if (percent < 0.05) return null
  const RADIAN = Math.PI / 180
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5
  const x = cx + radius * Math.cos(-midAngle * RADIAN)
  const y = cy + radius * Math.sin(-midAngle * RADIAN)
  return (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central"
      style={{ fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 700 }}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  )
}

// ── Stat card ──────────────────────────────────────────────────────────────────
function StatCard({ label, value, color, dark, accent }) {
  const bg = dark ? '#111C30' : 'white'
  const border = dark ? '#1E2D45' : '#EEF0F6'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div
      style={{ backgroundColor: bg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${border}`, flex: 1, position: 'relative', overflow: 'hidden', transition: 'transform 0.18s, box-shadow 0.18s' }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = dark ? '0 12px 32px rgba(0,0,0,0.3)' : '0 12px 32px rgba(15,35,71,0.1)' }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none' }}>
      {/* top accent line */}
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: accent || `linear-gradient(90deg, ${color}, transparent)` }} />
      <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.12em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>{label}</div>
      <div style={{ fontSize: '2rem', fontWeight: 800, color: color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{value}</div>
    </div>
  )
}

export default function ClientStats() {
  const navigate = useNavigate()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    api.get('/claims/my')
      .then(res => setClaims(extractClaims(res.data)))
      .catch(err => console.error('ClientStats fetch error:', err))
      .finally(() => setLoading(false))
  }, [])

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const gridColor = dark ? '#1E2D45' : '#F3F4F6'
  const gold = '#C9A84C'
  const navy = '#0F2347'

  const stats = {
    total: claims.length,
    approved: claims.filter(c => c.status === 'APPROVED').length,
    rejected: claims.filter(c => c.status === 'REJECTED').length,
    avgScore: (() => {
      const scored = claims.filter(c => c.analysis?.finalScore != null)
      return scored.length > 0
        ? Math.round(scored.reduce((s, c) => s + c.analysis.finalScore, 0) / scored.length)
        : 0
    })(),
    totalAmount: claims.reduce((s, c) => s + (c.claimedAmount || 0), 0),
  }

  const pieData = Object.entries(
    claims.reduce((acc, c) => {
      acc[c.status] = (acc[c.status] || 0) + 1
      return acc
    }, {})
  ).map(([status, count]) => ({
    name: STATUS_LABELS[status] || status,
    value: count,
    color: STATUS_COLORS[status] || '#9CA3AF',
  }))

  const scoreData = [
    { range: '0–20', count: claims.filter(c => (c.analysis?.finalScore ?? 50) <= 20).length, color: '#1A7A4A' },
    { range: '21–40', count: claims.filter(c => { const s = c.analysis?.finalScore ?? 50; return s > 20 && s <= 40 }).length, color: '#27AE60' },
    { range: '41–60', count: claims.filter(c => { const s = c.analysis?.finalScore ?? 50; return s > 40 && s <= 60 }).length, color: '#F39C12' },
    { range: '61–80', count: claims.filter(c => { const s = c.analysis?.finalScore ?? 50; return s > 60 && s <= 80 }).length, color: '#E67E22' },
    { range: '81–100', count: claims.filter(c => (c.analysis?.finalScore ?? 50) > 80).length, color: '#C0392B' },
  ]

  const monthData = claims.reduce((acc, c) => {
    const month = new Date(c.incidentDate).toLocaleDateString('fr-FR', { month: 'short', year: '2-digit' })
    const existing = acc.find(a => a.month === month)
    if (existing) { existing.montant += c.claimedAmount || 0; existing.sinistres += 1 }
    else acc.push({ month, montant: c.claimedAmount || 0, sinistres: 1 })
    return acc
  }, []).slice(-6)

  const scoreEvolution = claims
    .filter(c => c.analysis?.finalScore != null)
    .map((c, i) => ({
      index: i + 1,
      score: Math.round(c.analysis.finalScore),
      reference: c.reference,
    }))

  // Section header helper
  const SectionHeader = ({ title, sub }) => (
    <div style={{ marginBottom: '1.25rem' }}>
      <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 700, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0, letterSpacing: '-0.01em' }}>{title}</h3>
      {sub && <p style={{ color: textSub, fontSize: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: '0.2rem 0 0' }}>{sub}</p>}
    </div>
  )

  const ChartCard = ({ children, style }) => (
    <div style={{
      backgroundColor: cardBg, borderRadius: 16, border: `1px solid ${cardBorder}`,
      padding: '1.5rem', position: 'relative', overflow: 'hidden', ...style
    }}>
      {/* subtle gradient corner */}
      <div style={{ position: 'absolute', top: 0, right: 0, width: 120, height: 120, background: dark ? 'radial-gradient(circle at top right, rgba(201,168,76,0.04), transparent 70%)' : 'radial-gradient(circle at top right, rgba(15,35,71,0.03), transparent 70%)', pointerEvents: 'none' }} />
      {children}
    </div>
  )

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <style>{`
          @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        `}</style>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Analyse</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Mes <strong>statistiques</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              Vue d'ensemble de vos sinistres et scores IA
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button
              onClick={toggleDark}
              style={{
                background: 'none',
                border: `1px solid ${dark ? 'rgba(201,168,76,0.35)' : 'rgba(15,35,71,0.15)'}`,
                borderRadius: 7, padding: '0.4rem 0.85rem', cursor: 'pointer',
                color: dark ? 'rgba(201,168,76,0.9)' : textSub,
                fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
                transition: 'border-color 0.2s, color 0.2s',
                backgroundColor: dark ? 'rgba(201,168,76,0.06)' : 'transparent',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = gold; e.currentTarget.style.color = gold }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = dark ? 'rgba(201,168,76,0.35)' : 'rgba(15,35,71,0.15)'; e.currentTarget.style.color = dark ? 'rgba(201,168,76,0.9)' : textSub }}
            >
              {dark ? 'Mode clair' : 'Mode sombre'}
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '4rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement des statistiques...</div>
        ) : claims.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '4rem', backgroundColor: cardBg, borderRadius: 16, border: `1px solid ${cardBorder}` }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
            <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Aucune donnée disponible</div>
            <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '1.5rem' }}>Soumettez des sinistres pour voir vos statistiques</div>
            <button onClick={() => navigate('/client/new-claim')}
              style={{ padding: '0.65rem 1.5rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
              + Nouveau sinistre
            </button>
          </div>
        ) : (
          <>
            {/* ── KPI strip ── */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', animation: 'fadeUp 0.6s ease both' }}>
              <StatCard dark={dark} label="Total sinistres" value={stats.total} color={dark ? 'white' : navy} accent={`linear-gradient(90deg, ${navy}, transparent)`} />
              <StatCard dark={dark} label="Approuvés" value={stats.approved} color="#1A7A4A" accent="linear-gradient(90deg, #1A7A4A, transparent)" />
              <StatCard dark={dark} label="Rejetés" value={stats.rejected} color="#C0392B" accent="linear-gradient(90deg, #C0392B, transparent)" />
              <StatCard dark={dark} label="Score moyen IA"
                value={`${stats.avgScore}/100`}
                color={stats.avgScore > 70 ? '#C0392B' : stats.avgScore > 30 ? '#F39C12' : '#1A7A4A'}
                accent={`linear-gradient(90deg, ${stats.avgScore > 70 ? '#C0392B' : stats.avgScore > 30 ? '#F39C12' : '#1A7A4A'}, transparent)`}
              />
              <StatCard dark={dark} label="Montant total"
                value={stats.totalAmount > 0 ? `${(stats.totalAmount / 1000000).toFixed(1)}M DA` : '0 DA'}
                color="#2E86C1"
                accent="linear-gradient(90deg, #2E86C1, transparent)"
              />
            </div>

            {/* ── Row 1: Pie + Score bars ── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem', animation: 'fadeUp 0.7s ease 0.1s both' }}>

              {/* Pie chart */}
              <ChartCard>
                <SectionHeader title="Répartition des statuts" sub="Distribution par état de traitement" />
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <defs>
                      {pieData.map((d, i) => (
                        <radialGradient key={i} id={`pieGrad${i}`} cx="50%" cy="50%" r="50%">
                          <stop offset="0%" stopColor={d.color} stopOpacity={0.9} />
                          <stop offset="100%" stopColor={d.color} stopOpacity={0.7} />
                        </radialGradient>
                      ))}
                    </defs>
                    <Pie
                      data={pieData}
                      cx="50%" cy="50%"
                      outerRadius={90}
                      innerRadius={44}
                      dataKey="value"
                      labelLine={false}
                      label={renderCustomLabel}
                      paddingAngle={2}
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={`url(#pieGrad${i})`} stroke={cardBg} strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null
                        const d = payload[0].payload
                        return (
                          <div style={{ backgroundColor: cardBg, border: `1px solid ${cardBorder}`, borderRadius: 10, padding: '0.65rem 0.9rem', boxShadow: '0 8px 24px rgba(0,0,0,0.15)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                              <div style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: d.color }} />
                              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{d.name}</span>
                            </div>
                            <div style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.2rem' }}>{d.value} sinistre(s)</div>
                          </div>
                        )
                      }}
                    />
                    <Legend
                      iconType="circle"
                      iconSize={8}
                      formatter={(value) => (
                        <span style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{value}</span>
                      )}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>

              {/* Score distribution bars */}
              <ChartCard>
                <SectionHeader title="Distribution des scores IA" sub="Nombre de sinistres par tranche de score" />
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={scoreData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }} barSize={32}>
                    <defs>
                      {scoreData.map((d, i) => (
                        <linearGradient key={i} id={`barGrad${i}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={d.color} stopOpacity={0.95} />
                          <stop offset="100%" stopColor={d.color} stopOpacity={0.55} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                    <XAxis dataKey="range" tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} allowDecimals={false} axisLine={false} tickLine={false} />
                    <Tooltip content={<CustomTooltip dark={dark} />} cursor={{ fill: dark ? 'rgba(255,255,255,0.03)' : 'rgba(15,35,71,0.03)', radius: 6 }} />
                    <Bar dataKey="count" name="Sinistres" radius={[6, 6, 0, 0]}>
                      {scoreData.map((entry, i) => (
                        <Cell key={i} fill={`url(#barGrad${i})`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            {/* ── Row 2: Area charts ── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem', animation: 'fadeUp 0.7s ease 0.2s both' }}>

              {/* Amounts over time */}
              <ChartCard>
                <SectionHeader title="Montants déclarés" sub="Évolution sur les derniers mois" />
                {monthData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={monthData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="gradMontant" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={navy} stopOpacity={dark ? 0.5 : 0.25} />
                          <stop offset="95%" stopColor={navy} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                      <XAxis dataKey="month" tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} tickFormatter={v => `${(v / 1000).toFixed(0)}K`} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip dark={dark} />} />
                      <Area type="monotone" dataKey="montant" name="Montant (DA)" stroke={navy} fill="url(#gradMontant)" strokeWidth={2.5} dot={{ fill: navy, r: 4, strokeWidth: 0 }} activeDot={{ r: 6, fill: gold, strokeWidth: 0 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>Pas assez de données</div>
                )}
              </ChartCard>

              {/* Score evolution */}
              <ChartCard>
                <SectionHeader title="Évolution des scores IA" sub="Score de fraude par sinistre soumis" />
                {scoreEvolution.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={scoreEvolution} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="gradScore" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={gold} stopOpacity={dark ? 0.45 : 0.25} />
                          <stop offset="95%" stopColor={gold} stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
                      <XAxis dataKey="reference" tick={{ fontSize: 8, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip dark={dark} />} />
                      {/* Threshold reference lines via custom background */}
                      <Area type="monotone" dataKey="score" name="Score IA" stroke={gold} fill="url(#gradScore)" strokeWidth={2.5} dot={{ fill: gold, r: 4, strokeWidth: 0 }} activeDot={{ r: 6, fill: '#E8C97A', strokeWidth: 0 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>Pas de scores disponibles</div>
                )}
              </ChartCard>
            </div>

            {/* ── Legend strip ── */}
            <div style={{
              backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`,
              padding: '0.9rem 1.5rem', display: 'flex', gap: '2rem', alignItems: 'center',
              animation: 'fadeUp 0.7s ease 0.3s both',
            }}>
              <span style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>Zones IA :</span>
              {[['0–29', '#1A7A4A', 'Auto approuvé'], ['30–69', '#F39C12', 'Révision humaine'], ['70–100', '#C0392B', 'Auto rejeté']].map(([r, c, l]) => (
                <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: 10, height: 10, borderRadius: 3, backgroundColor: c }} />
                  <span style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    <strong style={{ color: c }}>{r}</strong> — {l}
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}