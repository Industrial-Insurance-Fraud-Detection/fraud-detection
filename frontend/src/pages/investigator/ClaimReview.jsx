import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import { InvestigatorSidebar } from '../../components/layout/InvestigatorLayout'
import NotificationBell from '../../components/ui/NotificationBell'

/**
 * ClaimReview — Investigator
 * Feature 11: View any claim (full detail, client, files, AI scores)
 * Feature 12: View client profile (GET /users/:id — investigator only)
 * Feature 13: Submit decision (PATCH /claims/:id/decide)
 * Feature 18: Download any file (GET /files/:id/url)
 */

function clientName(client) {
  return `${client?.firstName || ''} ${client?.lastName || ''}`.trim() || 'Client'
}

function ScoreGauge({ score }) {
  const color = score > 70 ? '#C0392B' : score > 30 ? '#F39C12' : '#1A7A4A'
  const circ = 2 * Math.PI * 54
  const offset = circ - (score / 100) * circ
  return (
    <div style={{ textAlign: 'center', padding: '1.5rem' }}>
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <svg width={140} height={140} viewBox="0 0 140 140">
          <circle cx={70} cy={70} r={54} fill="none" stroke="#F3F4F6" strokeWidth={10} />
          <circle cx={70} cy={70} r={54} fill="none" stroke={color} strokeWidth={10}
            strokeDasharray={circ} strokeDashoffset={offset}
            strokeLinecap="round" transform="rotate(-90 70 70)" />
        </svg>
        <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{score}</div>
          <div style={{ fontSize: '0.65rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
      <div style={{ marginTop: '0.5rem', fontSize: '0.82rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
        {score > 70 ? 'Fraude tres probable' : score > 30 ? 'Zone grise — revision requise' : 'Sinistre credible'}
      </div>
    </div>
  )
}

// Feature 18 — download any file for investigator
function FileRow({ file, dark }) {
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState('')
  const cardBg = dark ? '#0D1626' : '#F9FAFB'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  const handleDownload = async () => {
    setDownloading(true); setDlError('')
    try {
      const res = await api.get(`/files/${file.id}/url`)
      const data = res.data?.data ?? res.data
      window.open(data.url, '_blank', 'noopener,noreferrer')
    } catch {
      setDlError('Erreur de telechargement')
    } finally {
      setDownloading(false)
    }
  }

  const icon = file.fileType === 'CSV' ? '📊' : file.fileType === 'PHOTO' ? '🖼' : '📄'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.6rem 0.9rem', backgroundColor: cardBg, borderRadius: 8, border: `1px solid ${cardBorder}` }}>
      <span style={{ fontSize: '1rem' }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 500, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{file.fileName}</div>
        <div style={{ fontSize: '0.68rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          {file.fileType}{file.fileSize ? ` — ${(file.fileSize / 1024).toFixed(1)} KB` : ''}
        </div>
        {dlError && <div style={{ fontSize: '0.65rem', color: '#C0392B' }}>{dlError}</div>}
      </div>
      <button onClick={handleDownload} disabled={downloading}
        style={{ padding: '0.35rem 0.8rem', background: downloading ? '#E5E7EB' : 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: downloading ? '#9CA3AF' : 'white', border: 'none', borderRadius: 6, fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: downloading ? 'not-allowed' : 'pointer' }}>
        {downloading ? '...' : '↓ Telecharger'}
      </button>
    </div>
  )
}

// Feature 12 — client profile panel (GET /users/:id — investigator only)
function ClientProfilePanel({ clientId, dark }) {
  const [profile, setProfile] = useState(null)
  const [loading, setLoading] = useState(true)
  const cardBg = dark ? '#0D1626' : '#F9FAFB'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  useEffect(() => {
    if (!clientId) return
    api.get(`/users/${clientId}`)
      .then(res => setProfile(res.data?.data ?? res.data))
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [clientId])

  if (loading) return <div style={{ padding: '1rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.82rem' }}>Chargement profil client...</div>
  if (!profile) return null

  const claimCount = profile._count?.claims ?? 0
  return (
    <div style={{ backgroundColor: cardBg, borderRadius: 10, border: `1px solid ${cardBorder}`, padding: '1.25rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <div style={{ width: 42, height: 42, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, fontSize: '1.1rem', flexShrink: 0 }}>
          {(profile.firstName?.[0] ?? 'C').toUpperCase()}
        </div>
        <div>
          <div style={{ fontSize: '0.9rem', fontWeight: 700, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{clientName(profile)}</div>
          <div style={{ fontSize: '0.72rem', color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{profile.company || 'Client'}</div>
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
        {[
          ['Email', profile.email],
          ['Telephone', profile.phone || '—'],
          ['Wilaya', profile.wilaya || '—'],
          ['Sinistres totaux', `${claimCount} sinistre(s)`],
        ].map(([k, v]) => (
          <div key={k} style={{ padding: '0.5rem 0.6rem', backgroundColor: dark ? '#111C30' : 'white', borderRadius: 6, border: `1px solid ${cardBorder}` }}>
            <div style={{ fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.15rem' }}>{k}</div>
            <div style={{ fontSize: '0.78rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', wordBreak: 'break-word' }}>{v}</div>
          </div>
        ))}
      </div>
      {profile.createdAt && (
        <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          Client depuis {new Date(profile.createdAt).toLocaleDateString('fr-FR', { month: 'long', year: 'numeric' })}
        </div>
      )}
    </div>
  )
}

export default function ClaimReview() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [dark, toggleDark] = useDarkMode()

  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [outcome, setOutcome] = useState('')
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [done, setDone] = useState(false)
  const [showClient, setShowClient] = useState(false)

  useEffect(() => {
    api.get(`/claims/${id}`)
      .then(res => setClaim(res.data?.data ?? res.data))
      .catch(() => setError('Dossier introuvable'))
      .finally(() => setLoading(false))
  }, [id])

  // Feature 13 — submit decision
  const handleDecision = async () => {
    if (!outcome || notes.trim().length < 10) return
    setSubmitting(true)
    try {
      await api.patch(`/claims/${id}/decide`, { outcome, notes: notes.trim() })
      setDone(true)
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la soumission')
    } finally {
      setSubmitting(false)
    }
  }

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'

  if (loading) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
        Chargement du dossier...
      </div>
    </div>
  )

  if (error || !claim) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ fontSize: '3rem' }}>🔍</div>
        <div style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{error || 'Dossier introuvable'}</div>
        <button onClick={() => navigate('/investigator/dashboard')} style={{ padding: '0.6rem 1.2rem', background: '#0F2347', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Retour au dashboard</button>
      </div>
    </div>
  )

  const cName = clientName(claim.client)
  const eqName = claim.equipment?.name || '-'
  const analysis = claim.analysis || {}
  const finalScore = Math.round(analysis.finalScore ?? 50)
  const anomalyScore = Math.round(analysis.anomalyScore ?? 50)
  const classificationScore = Math.round(analysis.classificationScore ?? 50)
  const nlpScore = Math.round(analysis.nlpScore ?? 50)
  const visionScore = Math.round(analysis.visionScore ?? 50)
  const invName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'

  if (done) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', backgroundColor: cardBg, borderRadius: 16, padding: '3rem', border: `1px solid ${cardBorder}`, maxWidth: 480, width: '100%' }}>
          <div style={{ width: 64, height: 64, borderRadius: '50%', backgroundColor: outcome === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `2px solid ${outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', margin: '0 auto 1.5rem' }}>
            {outcome === 'APPROVED' ? '✓' : '✕'}
          </div>
          <h2 style={{ color: textMain, fontSize: '1.4rem', fontWeight: 400, marginBottom: '0.5rem' }}>
            Dossier <strong>{outcome === 'APPROVED' ? 'Approuve' : 'Rejete'}</strong>
          </h2>
          <p style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.88rem', marginBottom: '0.5rem' }}>{claim.reference} — {cName}</p>
          <p style={{ color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem', marginBottom: '2rem', fontStyle: 'italic' }}>"{notes}"</p>
          <div style={{ backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 8, padding: '0.75rem', marginBottom: '2rem', fontSize: '0.8rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            Decision enregistree par <strong style={{ color: textMain }}>{invName}</strong> — {new Date().toLocaleString('fr-FR')}
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
            <button onClick={() => navigate('/investigator/dashboard')} style={{ padding: '0.75rem 1.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
              Retour au dashboard
            </button>
            <button onClick={() => navigate('/investigator/history')} style={{ padding: '0.75rem 1.5rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              Voir historique
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <button onClick={() => navigate('/investigator/dashboard')} style={{ background: 'none', border: 'none', color: textSub, fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '0.5rem' }}>
              ← Retour au dashboard
            </button>
            <h1 style={{ fontSize: '1.75rem', color: textMain, fontWeight: 400 }}>
              Revision dossier <strong>{claim.reference}</strong>
            </h1>
            <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {cName} — {eqName}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
            <div style={{ backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, padding: '0.5rem 1rem', fontSize: '0.82rem', color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>
              Revision humaine requise
            </div>
          </div>
        </div>

        {error && <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 6, padding: '0.7rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>{error}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem' }}>
          {/* Left column */}
          <div>
            {/* Claim info */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Informations du dossier</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                {[
                  ['Client', cName],
                  ['Montant reclame', claim.claimedAmount != null ? `${claim.claimedAmount.toLocaleString('fr-FR')} DA` : '—'],
                  ['Equipement', eqName],
                  ['Type equipement', claim.equipment?.type || '—'],
                  ['Date incident', new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Classe fraude IA', analysis.fraudClass || 'N/A'],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                    <div style={{ fontSize: '0.88rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Description</div>
                <div style={{ fontSize: '0.88rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.65, backgroundColor: dark ? '#0D1626' : '#F9FAFB', padding: '0.75rem', borderRadius: 8, border: `1px solid ${cardBorder}` }}>{claim.description}</div>
              </div>
            </div>

            {/* Feature 11 — AI scores per model */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Analyse par les 4 modeles IA</h2>
              {[
                ['Modele 1 — Anomalie capteurs (35%)', anomalyScore],
                ['Modele 2 — Classification panne (25%)', classificationScore],
                ['Modele 3 — Analyse NLP (20%)', nlpScore],
                ['Modele 4 — Vision photos (20%)', visionScore],
              ].map(([title, sc]) => {
                const color = sc > 70 ? '#C0392B' : sc > 30 ? '#F39C12' : '#1A7A4A'
                return (
                  <div key={title} style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', padding: '0.85rem', border: `1px solid ${cardBorder}`, borderRadius: 8, marginBottom: '0.6rem' }}>
                    <div style={{ textAlign: 'center', minWidth: 48 }}>
                      <div style={{ fontSize: '1.4rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{sc}</div>
                      <div style={{ fontSize: '0.6rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ height: 5, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden', marginBottom: '0.4rem' }}>
                        <div style={{ height: '100%', width: `${sc}%`, backgroundColor: color, borderRadius: 3 }} />
                      </div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{title}</div>
                      <div style={{ fontSize: '0.75rem', color, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, marginTop: 2 }}>
                        {sc > 70 ? 'Score eleve — suspect' : sc > 30 ? 'Zone grise' : 'Score normal'}
                      </div>
                    </div>
                  </div>
                )
              })}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 8 }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combine</span>
                <span style={{ fontSize: '1.2rem', fontWeight: 700, color: finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{finalScore} / 100</span>
              </div>
            </div>

            {/* Feature 11 — All files with download (Feature 18) */}
            {claim.files?.length > 0 && (
              <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>Fichiers joints ({claim.files.length})</h2>
                  <span style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Liens valides 15 minutes</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {claim.files.map(f => <FileRow key={f.id} file={f} dark={dark} />)}
                </div>
              </div>
            )}

            {/* Feature 12 — Client profile */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showClient ? '1rem' : 0 }}>
                <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>Profil du client</h2>
                <button onClick={() => setShowClient(s => !s)}
                  style={{ padding: '0.35rem 0.85rem', border: `1.5px solid ${cardBorder}`, borderRadius: 6, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
                  {showClient ? 'Masquer' : 'Afficher'}
                </button>
              </div>
              {showClient && <ClientProfilePanel clientId={claim.clientId} dark={dark} />}
            </div>
          </div>

          {/* Right column */}
          <div>
            {/* Score gauge */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, marginBottom: '1.5rem' }}>
              <ScoreGauge score={finalScore} />
              <div style={{ padding: '0 1.5rem 1rem' }}>
                <div style={{ backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, padding: '0.75rem', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>ZONE GRISE</div>
                  <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>Revision humaine obligatoire</div>
                </div>
              </div>
            </div>

            {/* Feature 13 — Decision panel */}
            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, padding: '1.5rem', position: 'sticky', top: '2rem' }}>
              <h2 style={{ color: textMain, fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Votre decision</h2>
              <p style={{ color: textSub, fontSize: '0.78rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>
                Decision definitive — le client sera notifie automatiquement
              </p>

              <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
                {[
                  { key: 'APPROVED', icon: '✓', label: 'Approuver', desc: 'Sinistre legitime', color: '#1A7A4A', bg: '#F0FAF4', border: '#B8E4CA' },
                  { key: 'REJECTED', icon: '✕', label: 'Rejeter', desc: 'Fraude probable', color: '#C0392B', bg: '#FDF2F2', border: '#EBCECE' },
                ].map(r => (
                  <div key={r.key} onClick={() => setOutcome(r.key)}
                    style={{ flex: 1, padding: '1rem', border: `2px solid ${outcome === r.key ? r.color : cardBorder}`, borderRadius: 10, cursor: 'pointer', textAlign: 'center', backgroundColor: outcome === r.key ? r.bg : dark ? '#0D1626' : '#F9FAFB', transition: 'all 0.2s' }}>
                    <div style={{ fontSize: '1.4rem', marginBottom: '0.3rem', color: outcome === r.key ? r.color : textSub }}>{r.icon}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: outcome === r.key ? r.color : '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r.label}</div>
                    <div style={{ fontSize: '0.7rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{r.desc}</div>
                  </div>
                ))}
              </div>

              <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: textSub, marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                Justification obligatoire (min 10 caracteres)
              </label>
              <textarea rows={4} placeholder="Justifiez votre decision en detail..."
                value={notes} onChange={e => setNotes(e.target.value)}
                style={{ width: '100%', padding: '0.72rem', border: `1.5px solid ${!notes && outcome ? '#C0392B' : cardBorder}`, borderRadius: 6, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', outline: 'none', backgroundColor: dark ? '#0D1626' : '#F9FAFB', color: textMain, boxSizing: 'border-box', resize: 'vertical' }} />
              {notes.length > 0 && notes.length < 10 && (
                <div style={{ fontSize: '0.7rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.2rem' }}>
                  {10 - notes.length} caractere(s) manquant(s)
                </div>
              )}

              <button onClick={handleDecision}
                disabled={!outcome || notes.trim().length < 10 || submitting}
                style={{
                  width: '100%', marginTop: '1rem', padding: '0.85rem',
                  background: !outcome || notes.trim().length < 10
                    ? '#E5E7EB'
                    : outcome === 'APPROVED'
                      ? 'linear-gradient(135deg, #1A7A4A, #27AE60)'
                      : 'linear-gradient(135deg, #C0392B, #E74C3C)',
                  color: !outcome || notes.trim().length < 10 ? '#9CA3AF' : 'white',
                  border: 'none', borderRadius: 8, fontSize: '0.86rem',
                  fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600,
                  cursor: !outcome || notes.trim().length < 10 || submitting ? 'not-allowed' : 'pointer',
                  letterSpacing: '0.05em', textTransform: 'uppercase',
                }}>
                {submitting ? 'Enregistrement...' : !outcome ? 'Choisissez une decision' : outcome === 'APPROVED' ? 'Confirmer — Approuver' : 'Confirmer — Rejeter'}
              </button>
              <p style={{ fontSize: '0.72rem', color: textSub, textAlign: 'center', marginTop: '0.75rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                Cette action est irreversible
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}