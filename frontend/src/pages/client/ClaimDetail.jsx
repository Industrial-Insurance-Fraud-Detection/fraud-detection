import { useState, useEffect } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'

/**
 * ClaimDetail
 *
 * Features:
 *  - View claim detail (19)
 *  - Download files via presigned URL (24) — GET /files/:id/url
 *
 * Claim shape:
 *   claim.equipment          { name, type, ... }
 *   claim.client             { id, firstName, lastName, ... }
 *   claim.files              ClaimFile[]  { id, fileName, fileType, fileSize, minioPath }
 *   claim.analysis           { finalScore, anomalyScore, classificationScore, nlpScore, visionScore, fraudClass }
 *   claim.decision           { outcome, notes, type, createdAt, investigator }
 *   claim.claimedAmount      number
 *   claim.pdfUrl             string | null
 */

const STATUS_CONFIG = {
  APPROVED: { label: 'Approuvé', bg: '#F0FAF4', color: '#1A7A4A', border: '#B8E4CA' },
  REJECTED: { label: 'Rejeté', bg: '#FDF2F2', color: '#C0392B', border: '#EBCECE' },
  PENDING: { label: 'En attente', bg: '#FEF9E7', color: '#7D6608', border: '#F7DC6F' },
  ANALYZING: { label: 'Analyse en cours', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
  HUMAN_REVIEW: { label: 'Révision humaine', bg: '#EBF5FB', color: '#1A5276', border: '#AED6F1' },
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
      <div style={{ marginTop: '0.75rem' }}>
        <div style={{ fontSize: '0.88rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          {score > 70 ? 'Fraude très probable' : score > 30 ? 'Zone grise' : 'Sinistre crédible'}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>Score global de fraude</div>
      </div>
    </div>
  )
}

function AIModelCard({ title, score, weight }) {
  const s = Math.round(score)
  const color = s > 70 ? '#C0392B' : s > 30 ? '#F39C12' : '#1A7A4A'
  const label = s > 70 ? 'Score élevé — suspect' : s > 30 ? 'Zone grise' : 'Score normal'
  return (
    <div style={{ backgroundColor: 'white', border: '1px solid #EEF0F6', borderRadius: 10, padding: '1rem', marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{title}</div>
          <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Poids : {weight}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '1.4rem', fontWeight: 700, color, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{s}</div>
          <div style={{ fontSize: '0.65rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>/100</div>
        </div>
      </div>
      <div style={{ height: 6, backgroundColor: '#F3F4F6', borderRadius: 3, overflow: 'hidden', marginBottom: '0.5rem' }}>
        <div style={{ height: '100%', width: `${s}%`, backgroundColor: color, borderRadius: 3 }} />
      </div>
      <div style={{ fontSize: '0.78rem', fontWeight: 600, color, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{label}</div>
    </div>
  )
}

// ── File row with download button ─────────────────────────────────────────────
function FileRow({ file }) {
  const [downloading, setDownloading] = useState(false)
  const [dlError, setDlError] = useState('')

  const handleDownload = async () => {
    setDownloading(true)
    setDlError('')
    try {
      // GET /files/:id/url  →  { url, fileName, fileType, fileSize, expiresIn }
      const res = await api.get(`/files/${file.id}/url`)
      const data = res.data?.data ?? res.data
      // Open the presigned URL in a new tab for download
      window.open(data.url, '_blank', 'noopener,noreferrer')
    } catch (err) {
      setDlError('Erreur de téléchargement')
      console.error('Download error:', err)
    } finally {
      setDownloading(false)
    }
  }

  const icon = file.fileType === 'CSV' ? '📊' : file.fileType === 'PHOTO' ? '🖼' : '📄'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.6rem 0.9rem', backgroundColor: '#F9FAFB', borderRadius: 8, border: '1px solid #EEF0F6' }}>
      <span style={{ fontSize: '1rem' }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '0.82rem', fontWeight: 500, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{file.fileName}</div>
        <div style={{ fontSize: '0.68rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          {file.fileType} — {file.fileSize ? `${(file.fileSize / 1024).toFixed(1)} KB` : ''}
        </div>
        {dlError && <div style={{ fontSize: '0.65rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{dlError}</div>}
      </div>
      <button
        onClick={handleDownload}
        disabled={downloading}
        title="Télécharger le fichier (lien valide 15 minutes)"
        style={{ padding: '0.35rem 0.8rem', background: downloading ? '#E5E7EB' : 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: downloading ? '#9CA3AF' : 'white', border: 'none', borderRadius: 6, fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: downloading ? 'not-allowed' : 'pointer', transition: 'all 0.15s', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        {downloading ? '...' : '↓ Télécharger'}
      </button>
    </div>
  )
}

export default function ClaimDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const [claim, setClaim] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchClaim = () =>
    api.get(`/claims/${id}`)
      .then(res => {
        const data = res.data?.data ?? res.data
        setClaim(data)
      })
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

  if (loading) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', alignItems: 'center', justifyContent: 'center', fontFamily: 'Helvetica Neue, Arial, sans-serif', color: '#9CA3AF' }}>
      Chargement...
    </div>
  )

  if (error || !claim) return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#F7F8FC', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ fontSize: '3rem' }}>404</div>
      <div style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{error || 'Sinistre introuvable'}</div>
      <button onClick={() => navigate('/client/claims')}
        style={{ padding: '0.6rem 1.2rem', background: '#0F2347', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
        Retour
      </button>
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
    <div style={{ minHeight: '100vh', backgroundColor: '#F7F8FC', fontFamily: 'Georgia, serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '2rem' }}>

        {/* ── Page header ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <button onClick={() => navigate('/client/claims')}
              style={{ background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '0.5rem' }}>
              ← Retour aux sinistres
            </button>
            <h1 style={{ fontSize: '1.75rem', color: '#0F2347', fontWeight: 400 }}>
              Sinistre <strong>{claim.reference}</strong>
            </h1>
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
              {equipmentName} — {new Date(claim.incidentDate).toLocaleDateString('fr-FR')}
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            {claim.pdfUrl && (
              <a href={claim.pdfUrl} target="_blank" rel="noreferrer"
                style={{ padding: '0.5rem 1.1rem', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', color: '#0F2347', border: 'none', borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 700, cursor: 'pointer', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
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
          <div style={{ backgroundColor: '#EBF5FB', border: '1px solid #AED6F1', borderRadius: 10, padding: '1rem 1.5rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
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
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Informations du sinistre</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                {[
                  ['Équipement', equipmentName],
                  ['Type', equipmentType],
                  ['Montant réclamé', claim.claimedAmount != null ? `${claim.claimedAmount.toLocaleString('fr-FR')} DA` : '—'],
                  ['Date incident', new Date(claim.incidentDate).toLocaleDateString('fr-FR')],
                  ['Date soumission', new Date(claim.createdAt).toLocaleDateString('fr-FR')],
                  ['Référence', claim.reference],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.2rem' }}>{k}</div>
                    <div style={{ fontSize: '0.9rem', color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: '1rem' }}>
                <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Description</div>
                <div style={{ fontSize: '0.88rem', color: '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.6 }}>{claim.description}</div>
              </div>
            </div>

            {/* Files — with download buttons */}
            {claim.files?.length > 0 && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', margin: 0 }}>
                    Fichiers joints ({claim.files.length})
                  </h2>
                  <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    Liens valides 15 minutes
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {claim.files.map(f => (
                    <FileRow key={f.id} file={f} />
                  ))}
                </div>
              </div>
            )}

            {/* AI Analysis */}
            {analysis && finalScore != null && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem', marginBottom: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>
                  Analyse par les 4 modèles IA
                </h2>
                {anomalyScore != null && <AIModelCard title="Modèle 1 — Anomalie capteurs" score={anomalyScore} weight="35%" />}
                {classificationScore != null && <AIModelCard title="Modèle 2 — Classification panne" score={classificationScore} weight="25%" />}
                {nlpScore != null && <AIModelCard title="Modèle 3 — Analyse rapport NLP" score={nlpScore} weight="20%" />}
                {visionScore != null && <AIModelCard title="Modèle 4 — Vérification photos" score={visionScore} weight="20%" />}
                <div style={{ backgroundColor: '#F7F8FC', border: '1px solid #EEF0F6', borderRadius: 8, padding: '0.75rem 1rem', marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Score final combiné</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: 700, color: finalScore > 70 ? '#C0392B' : finalScore > 30 ? '#F39C12' : '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    {Math.round(finalScore)} / 100
                  </span>
                </div>
                {fraudClass && (
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    Classe de fraude : <strong style={{ color: '#0F2347' }}>{fraudClass}</strong>
                  </div>
                )}
              </div>
            )}

            {/* Decision */}
            {decision && (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem' }}>
                <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>
                  Décision finale
                  {investigatorName && (
                    <span style={{ fontWeight: 400, color: '#9CA3AF', fontSize: '0.82rem', marginLeft: '0.5rem' }}>
                      — {investigatorName}
                    </span>
                  )}
                </h2>
                <div style={{ marginBottom: '0.75rem' }}>
                  <span style={{ padding: '0.3rem 0.8rem', borderRadius: 20, fontSize: '0.78rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: decision.outcome === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', color: decision.outcome === 'APPROVED' ? '#1A7A4A' : '#C0392B', border: `1px solid ${decision.outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}` }}>
                    {decision.outcome === 'APPROVED' ? '✓ Approuvé' : '✕ Rejeté'}
                    {decision.type === 'AUTO' ? ' (automatique)' : ' (humain)'}
                  </span>
                </div>
                {investigatorNotes && (
                  <div style={{ backgroundColor: decision.outcome === 'APPROVED' ? '#F0FAF4' : '#FDF2F2', border: `1px solid ${decision.outcome === 'APPROVED' ? '#B8E4CA' : '#EBCECE'}`, borderRadius: 8, padding: '0.75rem 1rem', fontSize: '0.88rem', color: decision.outcome === 'APPROVED' ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontStyle: 'italic' }}>
                    "{investigatorNotes}"
                  </div>
                )}
                {decidedAt && (
                  <div style={{ fontSize: '0.75rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.5rem' }}>
                    Décision prise le {new Date(decidedAt).toLocaleString('fr-FR')}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Right column ── */}
          <div>
            {finalScore != null ? (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', marginBottom: '1.5rem' }}>
                <ScoreGauge score={Math.round(finalScore)} />
              </div>
            ) : (
              <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '2rem', marginBottom: '1.5rem', textAlign: 'center' }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⏳</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.5rem' }}>Analyse en cours</div>
                <div style={{ fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Le score IA sera disponible dans quelques secondes</div>
              </div>
            )}

            {/* Timeline */}
            <div style={{ backgroundColor: 'white', borderRadius: 12, border: '1px solid #EEF0F6', padding: '1.5rem' }}>
              <h2 style={{ color: '#0F2347', fontSize: '1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>Historique</h2>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {[
                  { label: 'Sinistre soumis', date: claim.createdAt, done: true },
                  { label: 'Analyse IA lancée', date: claim.createdAt, done: true },
                  { label: 'Analyse terminée', date: claim.updatedAt, done: finalScore != null },
                  { label: 'Décision finale', date: decidedAt, done: !!decidedAt },
                ].map((t, i, arr) => (
                  <div key={i} style={{ display: 'flex', gap: '0.75rem', paddingBottom: i < arr.length - 1 ? '1rem' : 0 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: t.done ? '#1A7A4A' : '#E5E7EB', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                        {t.done && <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'white' }} />}
                      </div>
                      {i < arr.length - 1 && <div style={{ width: 2, flex: 1, backgroundColor: t.done ? '#B8E4CA' : '#E5E7EB', marginTop: 4 }} />}
                    </div>
                    <div style={{ paddingBottom: i < arr.length - 1 ? '0.5rem' : 0 }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#0F2347', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{t.label}</div>
                      {t.date && <div style={{ fontSize: '0.72rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{new Date(t.date).toLocaleString('fr-FR')}</div>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}