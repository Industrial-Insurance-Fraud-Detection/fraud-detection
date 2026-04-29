import { useState, useEffect } from 'react'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import { InvestigatorSidebar } from '../../components/layout/InvestigatorLayout'
import NotificationBell from '../../components/ui/NotificationBell'

function extractArray(data) {
  const inner = data?.data ?? data
  const arr = inner?.data ?? inner
  return Array.isArray(arr) ? arr : []
}

function BarChart({ data, dark }) {
  const max = Math.max(...data.map(d => d.value), 1)
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  return (
    <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
      <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
        Distribution des scores IA (dossiers en revision)
      </h3>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.5rem', height: 160 }}>
        {data.map((d, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.4rem' }}>
            <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{d.value}</div>
            <div style={{ width: '100%', backgroundColor: d.color, borderRadius: '4px 4px 0 0', height: `${(d.value / max) * 120}px`, minHeight: 4, transition: 'height 0.6s ease' }} />
            <div style={{ fontSize: '0.65rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', textAlign: 'center' }}>{d.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function InvestigatorStats() {
  const [claims, setClaims] = useState([])
  const [loading, setLoading] = useState(true)
  const [dark, toggleDark] = useDarkMode()

  useEffect(() => {
    api.get('/claims/flagged?limit=100')
      .then(res => setClaims(extractArray(res.data)))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  const total = claims.length
  const urgent = claims.filter(c => (c.analysis?.finalScore ?? 50) >= 60).length
  const avgScore = total > 0 ? Math.round(claims.reduce((s, c) => s + (c.analysis?.finalScore ?? 50), 0) / total) : 0
  const totalAmount = claims.reduce((s, c) => s + (c.claimedAmount ?? 0), 0)

  const scoreDistrib = [
    { label: '0-20', value: claims.filter(c => (c.analysis?.finalScore ?? 50) <= 20).length, color: '#1A7A4A' },
    { label: '21-40', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 20 && (c.analysis?.finalScore ?? 50) <= 40).length, color: '#27AE60' },
    { label: '41-60', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 40 && (c.analysis?.finalScore ?? 50) <= 60).length, color: '#F39C12' },
    { label: '61-80', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 60 && (c.analysis?.finalScore ?? 50) <= 80).length, color: '#E67E22' },
    { label: '81-100', value: claims.filter(c => (c.analysis?.finalScore ?? 50) > 80).length, color: '#C0392B' },
  ]

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Analyse</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Statistiques <strong>dossiers en revision</strong></h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>Donnees issues des sinistres en cours d'examen humain</p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '3rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Chargement...</div>
        ) : (
          <>
            {/* Stat cards */}
            <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
              {[
                { label: 'Dossiers en revision', value: total, sub: 'Statut HUMAN_REVIEW', color: dark ? 'white' : '#0F2347' },
                { label: 'Urgents (score >= 60)', value: urgent, sub: 'Priorite haute', color: '#C0392B' },
                { label: 'Score IA moyen', value: `${avgScore}/100`, sub: 'Sur dossiers en cours', color: '#1A5276' },
                { label: 'Montant total', value: totalAmount > 0 ? `${(totalAmount / 1000000).toFixed(1)}M DA` : '0 DA', sub: 'Valeur en revision', color: '#2E86C1' },
              ].map(s => (
                <div key={s.label} style={{ backgroundColor: cardBg, borderRadius: 14, padding: '1.5rem', border: `1px solid ${cardBorder}`, flex: 1, transition: 'transform 0.18s' }}
                  onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
                  onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
                  <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.6rem' }}>{s.label}</div>
                  <div style={{ fontSize: '2.2rem', fontWeight: 700, color: s.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s.value}</div>
                  <div style={{ fontSize: '0.75rem', color: textSub, marginTop: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{s.sub}</div>
                </div>
              ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
              <BarChart data={scoreDistrib} dark={dark} />

              {/* Decision thresholds */}
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
                <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Seuils de decision automatique</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {[
                    { range: '0 – 29', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA', label: 'Auto-approuve', desc: 'Score faible — aucun indicateur de fraude' },
                    { range: '30 – 69', color: '#F39C12', bg: '#FEF9E7', border: '#F7DC6F', label: 'Revision humaine', desc: 'Zone grise — votre expertise est requise' },
                    { range: '70 – 100', color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE', label: 'Auto-rejete', desc: 'Score eleve — fraude probable' },
                  ].map(item => (
                    <div key={item.range} style={{ display: 'flex', gap: '1rem', padding: '0.75rem', backgroundColor: item.bg, border: `1px solid ${item.border}`, borderRadius: 8 }}>
                      <div style={{ fontWeight: 700, color: item.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.9rem', minWidth: 70 }}>{item.range}</div>
                      <div>
                        <div style={{ fontWeight: 600, color: item.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>{item.label}</div>
                        <div style={{ color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.75rem', marginTop: 2 }}>{item.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* AI models */}
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
              <h3 style={{ color: textMain, fontSize: '0.95rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Modeles IA du systeme</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                {[
                  { label: 'Modele 1', name: 'Isolation Forest + LSTM', poids: '35%' },
                  { label: 'Modele 2', name: 'XGBoost Classification', poids: '25%' },
                  { label: 'Modele 3', name: 'BERT NLP multilingue', poids: '20%' },
                  { label: 'Modele 4', name: 'YOLOv8 + ELA Vision', poids: '20%' },
                ].map(m => (
                  <div key={m.label} style={{ padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}` }}>
                    <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>{m.label} — {m.poids}</div>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.4rem' }}>{m.name}</div>
                    <span style={{ padding: '0.2rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: '#F0FAF4', color: '#1A7A4A', border: '1px solid #B8E4CA' }}>Actif</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}