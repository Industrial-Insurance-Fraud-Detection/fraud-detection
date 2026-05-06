import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import Sidebar from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuvé', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED: { label: 'Rejeté', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
  HUMAN_REVIEW: { label: 'Révision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
}

function ScoreGauge({ score, dark }) {
  const color = score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A'
  const circ = 2 * Math.PI * 54
  const offset = circ - (score / 100) * circ
  const trackColor = dark ? '#1E2D45' : '#F3F4F6'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  return (
    <div style={{ textAlign: 'center', padding: '2rem 1.5rem 1.5rem' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        {/* outer glow ring */}
        <div style={{
          position: 'absolute', inset: -8, borderRadius: '50%',
          background: `radial-gradient(circle, ${color}18 0%, transparent 70%)`,
          pointerEvents: 'none',
        }} />
        <svg width={148} height={148} viewBox="0 0 148 148">
          <circle cx={74} cy={74} r={54} fill="none" stroke={trackColor} strokeWidth={10} />
          <circle cx={74} cy={74} r={54} fill="none" stroke={color} strokeWidth={10}
            strokeDasharray={circ} strokeDashoffset={offset}
            strokeLinecap="round" transform="rotate(-90 74 74)"
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.22,0.61,0.36,1)' }} />
        </svg>
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', textAlign: 'center' }}>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: '0.62rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.06em' }}>/100</div>
        </div>
      </div>
      <div style={{ marginTop: '1rem' }}>
        <div style={{
          display: 'inline-block', padding: '0.3rem 1rem', borderRadius: 20,
          backgroundColor: `${color}15`, border: `1px solid ${color}40`,
          fontSize: '0.82rem', fontWeight: 700, color,
          fontFamily: 'Helvetica Neue, Arial, sans-serif',
        }}>
          {score > 70 ? 'Fraude très probable' : score > 30 ? 'Zone grise' : 'Sinistre crédible'}
        </div>
        <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.4rem' }}>Score global de fraude</div>
      </div>
    </div>
  )
}

function AIModelCard({ title, score, weight, dark }) {
  const s = Math.round(score)
  const color = s > 70 ? '#C0392B' : s > 30 ? '#F39C12' : '#1A7A4A'
  const label = s > 70 ? 'Score élevé — suspect' : s > 30 ? 'Zone grise' : 'Score normal'
  const cardBg = dark ? '#0D1626' : '#F9FAFB'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const trackColor = dark ? '#1E2D45' : '#F3F4F6'

  return (
    <div style={{
      backgroundColor: cardBg, border: `1px solid ${cardBorder}`,
      borderRadius: 10, padding: '0.9rem 1rem', marginBottom: '0.65rem',
      borderLeft: `3px solid ${color}`,
      transition: 'transform 0.15s',
    }}
      onMouseEnter={e => e.currentTarget.style.transform = 'translateX(3px)'}
      onMouseLeave={e => e.currentTarget.style.transform = 'translateX(0)'}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{title}</div>
          <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>Poids : {weight}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s}</div>
          <div style={{ fontSize: '0.6rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
      <div style={{ height: 5, backgroundColor: trackColor, borderRadius: 3, overflow: 'hidden', marginBottom: '0.4rem' }}>
        <div style={{ height: '100%', width: `${s}%`, backgroundColor: color, borderRadius: 3, transition: 'width 0.8s ease' }} />
      </div>
      <div style={{ fontSize: '0.72rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{label}</div>
    </div>
  )
}

function FileRow({ file, dark }) {
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState('')
  const cardBg = dark ? '#0D1626' : '#F9FAFB'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  const handleDownload = async () => {
    setDownloading(true)
    setDlError('')
    try {
      const res = await api.get(`/files/${file.id}/url`)
      const data = res.data?.data ?? res.data
      window.open(data.url, '_blank', 'noopener,noreferrer')
    } catch (err) {
      setDlError('Erreur de téléchargement')
    } finally {
      setDownloading(false)
    }
  }

  const icon = file.fileType === 'CSV' ? '📊' : file.fileType === 'PHOTO' ? '🖼' : '📄'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.75rem',
      padding: '0.7rem 0.9rem', backgroundColor: cardBg,
      borderRadius: 8, border: `1px solid ${cardBorder}`,
      transition: 'border-color 0.15s',
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = '#C9A84C'}
      onMouseLeave={e => e.currentTarget.style.borderColor = cardBorder}
    >
      <div style={{ width: 36, height: 36, borderRadius: 8, backgroundColor: dark ? '#111C30' : 'white', border: `1px solid ${cardBorder}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem', flexShrink: 0 }}>
        {icon}
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 500, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{file.fileName}</div>
        <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>
          {file.fileType}{file.fileSize ? ` — ${(file.fileSize / 1024).toFixed(1)} KB` : ''}
        </div>
        {dlError && <div style={{ fontSize: '0.65rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{dlError}</div>}
      </div>
      <button
        onClick={handleDownload}
        disabled={downloading}
        style={{
          padding: '0.38rem 0.9rem',
          background: downloading ? (dark ? '#1E2D45' : '#E5E7EB') : 'linear-gradient(135deg, #0F2347, #1A3A6B)',
          color: downloading ? textSub : 'white',
          border: 'none', borderRadius: 6, fontSize: '0.72rem',
          fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600,
          cursor: downloading ? 'not-allowed' : 'pointer',
          transition: 'all 0.15s', display: 'flex', alignItems: 'center', gap: '0.3rem',
          whiteSpace: 'nowrap',
        }}>
        {downloading ? '...' : '↓ Télécharger'}
      </button>
    </div>
  )
}

export default function ClaimDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [dark, toggleDark] = useDarkMode()
  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchClaim = () =>
    api.get(`/claims/${id}`)
      .then(res => { const data = res.data?.data ?? res.data; setClaim(data) })
      .catch(err => {
        if (err.response?.status === 404) setError('Sinistre introuvable')
        else setError('Erreur lors du chargement')
      })
      .finally(() => setLoading(false))

  useEffect(() => { fetchClaim() }, [id])

  useEffect(() => {
    if (!claim) return
    if (!['ANALYZING', 'PENDING'].includes(claim.status)) return
    const interval = setInterval(fetchClaim, 5000)
    return () => clearInterval(interval)
  }, [claim?.status])

  // ── theme vars ──
  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const textBody = dark ? '#C8D8E8' : '#4B5563'
  const gold = '#C9A84C'
  const navy = '#0F2347'

  if (loading) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
        Chargement...
      </div>
    </div>
  )

  if (error || !claim) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg }}>
      <Sidebar role="CLIENT" dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ fontSize: '3rem', color: textSub }}>404</div>
        <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{error || 'Sinistre introuvable'}</div>
        <button onClick={() => navigate('/client/claims')}
          style={{ padding: '0.6rem 1.2rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          Retour
        </button>
      </div>
    </div>
  )

  const sc = STATUS_CONFIG[claim.status] || STATUS_CONFIG['PENDING']
  const analysis = claim.analysis || null
  const finalScore = analysis?.finalScore ?? null
  const anomalyScore = analysis?.anomalyScore ?? null
  const classificationScore = analysis?.classificationScore ?? null
  const nlpScore = analysis?.nlpScore ?? null
  const visionScore = analysis?.visionScore ?? null
  const fraudClass = analysis?.fraudClass ?? null
  const decision = claim.decision || null
  const investigatorNotes = decision?.notes ?? null
  const decidedAt = decision?.createdAt ?? null
  const investigatorName = decision?.investigator
    ? `${decision.investigator.firstName || ''} ${decision.investigator.lastName || ''}`.trim()
    : decision?.type === 'AUTO' ? 'Système IA (automatique)' : null
  const equipmentName = claim.equipment?.name || '—'
  const equipmentType = claim.equipment?.type || '—'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <Sidebar role="CLIENT" dark={dark} />

      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        {/* ── Page header ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <button onClick={() => navigate('/client/claims')}
              style={{ background: 'none', border: 'none', color: textSub, fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.3rem', transition: 'color 0.15s' }}
              onMouseEnter={e => e.currentTarget.style.color = gold}
              onMouseLeave={e => e.currentTarget.style.color = textSub}>
              ← Retour aux sinistres
            </button>
            <h1 style={{ fontSize: '1.75rem', color: textMain, fontWeight: 400, letterSpacing: '-0.02em' }}>
              Sinistre <strong style={{ color: gold }}>{claim.reference}</strong>
            </h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {equipmentName} — {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />

            {/* Landing-page style dark mode toggle */}
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

            {claim.pdfUrl && (
              <a href={claim.pdfUrl} target="_blank" rel="noreferrer"
                style={{ padding: '0.5rem 1.1rem', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', color: navy, border: 'none', borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 700, cursor: 'pointer', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.4rem', boxShadow: '0 4px 14px rgba(201,168,76,0.3)' }}>
                📄 Lettre de décision
              </a>
            )}

            <span style={{ padding: '0.4rem 1rem', borderRadius: 20, fontSize: '0.82rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: sc.bg, color: sc.color, border: `1px solid ${sc.border}` }}>
              {sc.label}
            </span>
          </div>
        </div>

        {/* Analyzing banner */}
        {['ANALYZING', 'PENDING'].includes(claim.status) && (
          <div style={{ backgroundColor: dark ? '#0D1E2B' : '#EBF5FB', border: '1px solid #AED6F1', borderRadius: 10, padding: '1rem 1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ fontSize: '1.5rem' }}>⏳</div>
            <div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#1A5276', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Analyse IA en cours...</div>
              <div style={{ fontSize: '0.8rem', color: '#5D9CEC', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Cette page se rafraîchit automatiquement toutes les 5 secondes</div>
            </div>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>

          {/* ── Left column ── */}
          <div>

            {/* Claim info card */}
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem', position: 'relative', overflow: 'hidden' }}>
              {/* top accent */}
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${navy}, #1A3A6B, transparent)` }} />
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>
                Informations du sinistre
              </h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                {[
                  ['Équipement', equipmentName],
                  ['Type', equipmentType],
                  ['Montant réclamé', claim.claimedAmount != null ? `${claim.claimedAmount.toLocaleString('fr-FR')} DA` : '—'],
                  ['Date incident', new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Date soumission', new Date(claim.createdAt).toLocaleDateString('fr-FR')],
                  ['Référence', claim.reference],
                ].map(([k, v]) => (
                  <div key={k} style={{ backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderRadius: 8, padding: '0.65rem 0.85rem', border: `1px solid ${cardBorder}` }}>
                    <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                    <div style={{ fontSize: '0.88rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ backgroundColor: dark ? '#0D1626' : '#F9FAFB', borderRadius: 8, padding: '0.75rem 0.85rem', border: `1px solid ${cardBorder}` }}>
                <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.35rem' }}>Description</div>
                <div style={{ fontSize: '0.88rem', color: textBody, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.65 }}>{claim.description}</div>
              </div>
            </div>

            {/* Files */}
            {claim.files?.length > 0 && (
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>
                    Fichiers joints <span style={{ color: gold }}>({claim.files.length})</span>
                  </h2>
                  <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#1A7A4A', display: 'inline-block' }} />
                    Liens valides 15 minutes
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {claim.files.map(f => <FileRow key={f.id} file={f} dark={dark} />)}
                </div>
              </div>
            )}

            {/* AI Analysis */}
            {analysis && finalScore != null && (
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'linear-gradient(90deg, #4A9EFF, #C9A84C, #1ABC9C, #9B59B6)' }} />
                <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>
                  Analyse par les 4 modèles IA
                </h2>
                {anomalyScore != null && <AIModelCard title="Modèle 1 — Anomalie capteurs" score={anomalyScore} weight="35%" dark={dark} />}
                {classificationScore != null && <AIModelCard title="Modèle 2 — Classification panne" score={classificationScore} weight="25%" dark={dark} />}
                {nlpScore != null && <AIModelCard title="Modèle 3 — Analyse rapport NLP" score={nlpScore} weight="20%" dark={dark} />}
                {visionScore != null && <AIModelCard title="Modèle 4 — Vérification photos" score={visionScore} weight="20%" dark={dark} />}

                {/* Final score row */}
                <div style={{
                  backgroundColor: dark ? '#0D1626' : '#F7F8FC', border: `1px solid ${cardBorder}`,
                  borderRadius: 8, padding: '0.85rem 1rem', marginTop: '0.75rem',
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                }}>
                  <span style={{ fontSize: '0.88rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combiné</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{ width: 80, height: 6, backgroundColor: dark ? '#1E2D45' : '#E5E7EB', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${finalScore}%`, backgroundColor: finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A', borderRadius: 3 }} />
                    </div>
                    <span style={{ fontSize: '1.25rem', fontWeight: 800, color: finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                      {Math.round(finalScore)} / 100
                    </span>
                  </div>
                </div>

                {fraudClass && (
                  <div style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    Classe de fraude :
                    <strong style={{ color: textMain, fontSize: '0.8rem' }}>{fraudClass}</strong>
                  </div>
                )}
              </div>
            )}

            {/* Decision */}
            {decision && (
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem', position: 'relative', overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: decision.outcome === 'APPROVED' ? 'linear-gradient(90deg, #1A7A4A, transparent)' : 'linear-gradient(90deg, #C0392B, transparent)' }} />
                <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.85rem' }}>
                  Décision finale
                  {investigatorName && (
                    <span style={{ fontWeight: 400, color: textSub, fontSize: '0.8rem', marginLeft: '0.5rem' }}>
                      — {investigatorName}
                    </span>
                  )}
                </h2>
                <div style={{ marginBottom: '0.85rem' }}>
                  <span style={{ padding: '0.3rem 0.9rem', borderRadius: 20, fontSize: '0.78rem', fontWeight: 700, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: decision.outcome === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', color: decision.outcome === 'APPROVED' ? '#1A7A4A' : '#C0392B', border: `1px solid ${decision.outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}` }}>
                    {decision.outcome === 'APPROVED' ? '✓ Approuvé' : '✕ Rejeté'}
                    {decision.type === 'AUTO' ? ' (automatique)' : ' (humain)'}
                  </span>
                </div>
                {investigatorNotes && (
                  <div style={{ backgroundColor: decision.outcome === 'APPROVED' ? (dark ? 'rgba(26,122,74,0.1)' : '#F0FAF4') : (dark ? 'rgba(192,57,43,0.1)' : '#FDF2F2'), border: `1px solid ${decision.outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, borderRadius: 8, padding: '0.85rem 1rem', fontSize: '0.88rem', color: decision.outcome === 'APPROVED' ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontStyle: 'italic', lineHeight: 1.6 }}>
                    "{investigatorNotes}"
                  </div>
                )}
                {decidedAt && (
                  <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.6rem' }}>
                    Décision prise le {new Date(decidedAt).toLocaleString('fr-FR')}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Right column ── */}
          <div>

            {/* Score gauge card */}
            {finalScore != null ? (
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, marginBottom: '1.5rem', overflow: 'hidden', position: 'relative' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, ${finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A'}, transparent)` }} />
                <ScoreGauge score={Math.round(finalScore)} dark={dark} />
              </div>
            ) : (
              <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '2rem', marginBottom: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>⏳</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>Analyse en cours</div>
                <div style={{ fontSize: '0.8rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Le score IA sera disponible dans quelques secondes</div>
              </div>
            )}

            {/* Timeline */}
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>Historique</h2>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {[
                  { label: 'Sinistre soumis', date: claim.createdAt, done: true },
                  { label: 'Analyse IA lancée', date: claim.createdAt, done: true },
                  { label: 'Analyse terminée', date: claim.updatedAt, done: finalScore != null },
                  { label: 'Décision finale', date: decidedAt, done: !!decidedAt },
                ].map((t, i, arr) => (
                  <div key={i} style={{ display: 'flex', gap: '0.85rem', paddingBottom: i < arr.length - 1 ? '1.1rem' : 0 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{
                        width: 18, height: 18, borderRadius: '50%', flexShrink: 0,
                        backgroundColor: t.done ? '#1A7A4A' : (dark ? '#1E2D45' : '#E5E7EB'),
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        boxShadow: t.done ? '0 0 0 3px rgba(26,122,74,0.15)' : 'none',
                        transition: 'all 0.3s',
                      }}>
                        {t.done && <div style={{ width: 7, height: 7, borderRadius: '50%', backgroundColor: 'white' }} />}
                      </div>
                      {i < arr.length - 1 && (
                        <div style={{ width: 2, flex: 1, backgroundColor: t.done ? '#B8E4CA' : (dark ? '#1E2D45' : '#E5E7EB'), marginTop: 4, borderRadius: 2 }} />
                      )}
                    </div>
                    <div style={{ paddingBottom: i < arr.length - 1 ? '0.5rem' : 0 }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: t.done ? textMain : textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{t.label}</div>
                      {t.date
                        ? <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{new Date(t.date).toLocaleString('fr-FR')}</div>
                        : <div style={{ fontSize: '0.7rem', color: dark ? '#2A3D55' : '#D1D5DB', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>En attente...</div>
                      }
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Score legend */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '0.85rem 1rem', marginTop: '1rem' }}>
              <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem', fontWeight: 600 }}>Zones de score IA</div>
              {[['0–29', '#1A7A4A', 'Auto approuvé'], ['30–69', '#F39C12', 'Révision humaine'], ['70–100', '#C0392B', 'Auto rejeté']].map(([r, c, l]) => (
                <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, backgroundColor: c, flexShrink: 0 }} />
                  <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    <strong style={{ color: c }}>{r}</strong> — {l}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}