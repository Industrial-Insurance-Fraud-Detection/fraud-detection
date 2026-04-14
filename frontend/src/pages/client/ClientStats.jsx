import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

const STATUS_COLORS = {
  APPROVED:     '#1A7A4A',
  REJECTED:     '#C0392B',
  HUMAN_REVIEW: '#F39C12',
  ANALYZING:    '#1A5276',
  PENDING:      '#7D6608',
}

const STATUS_LABELS = {
  APPROVED:     'Approuve',
  REJECTED:     'Rejete',
  HUMAN_REVIEW: 'Revision',
  ANALYZING:    'Analyse',
  PENDING:      'En attente',
}

function extractArray(data) {
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.items)) return data.items
  if (Array.isArray(data?.data)) return data.data
  if (Array.isArray(data?.data?.items)) return data.data.items
  return []
}

export default function ClientStats() {
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    api.get('/claims/my')
      .then(res => setClaims(extractArray(res.data)))
      .catch(err => console.error(err))
      .finally(() => setLoading(false))
  }, [])

  const pageBg    = dark ? '#0D1626' : '#F7F8FC'
  const cardBg    = dark ? '#111C30' : 'white'
  const cardBorder= dark ? '#1E2D45' : '#EEF0F6'
  const textMain  = dark ? 'white' : '#0F2347'
  const textSub   = dark ? '#5A7A9A' : '#9CA3AF'
  const gridColor = dark ? '#1E2D45' : '#F3F4F6'
  const tooltipBg = dark ? '#0D1626' : 'white'

  const pieData = Object.entries(
    claims.reduce((acc, c) => {
      acc[c.status] = (acc[c.status] || 0) + 1
      return acc
    }, {})
  ).map(([status, count]) => ({
    name:  STATUS_LABELS[status] || status,
    value: count,
    color: STATUS_COLORS[status] || '#9CA3AF'
  }))

  const scoreData = [
    { range: '0-20',  count: claims.filter(c => (c.finalScore||50) <= 20).length,  color: '#1A7A4A' },
    { range: '21-40', count: claims.filter(c => (c.finalScore||50) > 20 && (c.finalScore||50) <= 40).length, color: '#27AE60' },
    { range: '41-60', count: claims.filter(c => (c.finalScore||50) > 40 && (c.finalScore||50) <= 60).length, color: '#F39C12' },
    { range: '61-80', count: claims.filter(c => (c.finalScore||50) > 60 && (c.finalScore||50) <= 80).length, color: '#E67E22' },
    { range: '81-100',count: claims.filter(c => (c.finalScore||50) > 80).length, color: '#C0392B' },
  ]

  const monthData = claims.reduce((acc, c) => {
    const month = new Date(c.incidentDate).toLocaleDateString('fr-FR', { month: 'short', year: '2-digit' })
    const existing = acc.find(a => a.month === month)
    if (existing) {
      existing.montant += c.amount || 0
      existing.sinistres += 1
    } else {
      acc.push({ month, montant: c.amount || 0, sinistres: 1 })
    }
    return acc
  }, []).slice(-6)

  const scoreEvolution = claims
    .filter(c => c.finalScore !== null && c.finalScore !== undefined)
    .map((c, i) => ({
      index: i + 1,
      score: Math.round(c.finalScore),
      reference: c.reference,
    }))

  const stats = {
    total:       claims.length,
    approved:    claims.filter(c => c.status === 'APPROVED').length,
    rejected:    claims.filter(c => c.status === 'REJECTED').length,
    avgScore:    claims.length > 0 ? Math.round(claims.filter(c => c.finalScore).reduce((s, c) => s + c.finalScore, 0) / (claims.filter(c => c.finalScore).length || 1)) : 0,
    totalAmount: claims.reduce((s, c) => s + (c.amount || 0), 0),
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ backgroundColor: tooltipBg, border: `1px solid ${cardBorder}`, borderRadius: 8, padding: '0.75rem 1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>{label}</p>
          {payload.map((p, i) => (
            <p key={i} style={{ fontSize: '0.75rem', color: p.color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
              {p.name}: {typeof p.value === 'number' && p.value > 1000 ? `${(p.value/1000).toFixed(0)}K` : p.value}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

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
            <button onClick={toggleDark}
              style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '4rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement des statistiques...</div>
        ) : claims.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '4rem', backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}` }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
            <div style={{ color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Aucune donnee disponible</div>
            <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '1.5rem' }}>Soumettez des sinistres pour voir vos statistiques</div>
            <button onClick={() => navigate('/client/new-claim')}
              style={{ padding: '0.65rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
              + Nouveau sinistre
            </button>
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
              {[
                { label: 'Total sinistres', value: stats.total,       color: dark ? 'white' : '#0F2347' },
                { label: 'Approuves',       value: stats.approved,    color: '#1A7A4A' },
                { label: 'Rejetes',         value: stats.rejected,    color: '#C0392B' },
                { label: 'Score moyen IA',  value: `${stats.avgScore}/100`, color: stats.avgScore > 70 ? '#C0392B' : stats.avgScore > 30 ? '#F39C12' : '#1A7A4A' },
                { label: 'Montant total',   value: `${(stats.totalAmount/1000000).toFixed(1)}M DA`, color: '#2E86C1' },
              ].map(s => (
                <div key={s.label} style={{ backgroundColor: cardBg, borderRadius: 12, padding: '1.25rem', border: `1px solid ${cardBorder}`, flex: 1, transition: 'transform 0.18s' }}
                  onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
                  <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.4rem' }}>{s.label}</div>
                  <div style={{ fontSize: '1.8rem', fontWeight: 700, color: s.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s.value}</div>
                </div>
              ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                  Repartition des statuts
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: tooltipBg, border: `1px solid ${cardBorder}`, borderRadius: 8, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                  Distribution des scores IA
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={scoreData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="range" tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} />
                    <YAxis tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} allowDecimals={false} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" name="Sinistres" radius={[4, 4, 0, 0]}>
                      {scoreData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                  Montants declares par periode
                </h3>
                {monthData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={monthData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="colorMontant" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0F2347" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#0F2347" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis dataKey="month" tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} />
                      <YAxis tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} tickFormatter={v => `${(v/1000).toFixed(0)}K`} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="montant" name="Montant (DA)" stroke="#0F2347" fill="url(#colorMontant)" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>
                    Pas assez de donnees
                  </div>
                )}
              </div>

              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                  Evolution des scores IA
                </h3>
                {scoreEvolution.length > 0 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={scoreEvolution} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#C9A84C" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#C9A84C" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis dataKey="reference" tick={{ fontSize: 9, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="score" name="Score IA" stroke="#C9A84C" fill="url(#colorScore)" strokeWidth={2} dot={{ fill: '#C9A84C', r: 4 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>
                    Pas de scores disponibles
                  </div>
                )}
              </div>
            </div>

            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1rem 1.5rem', display: 'flex', gap: '2rem', alignItems: 'center' }}>
              <span style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>Zones de score IA :</span>
              {[['0-29', '#1A7A4A', 'Approuve automatiquement'], ['30-69', '#F39C12', 'Revision humaine'], ['70-100', '#C0392B', 'Rejete automatiquement']].map(([r, c, l]) => (
                <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: 12, height: 12, borderRadius: 3, backgroundColor: c }} />
                  <span style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r} — {l}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}