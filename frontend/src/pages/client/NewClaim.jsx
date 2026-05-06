import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import Sidebar, { useDarkMode } from '../../components/layout/Sidebar'

function StepIndicator({ current }) {
  const steps = ['Informations', 'Documents', 'Confirmation']
  return (
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2rem' }}>
      {steps.map((s, i) => (
        <div key={s} style={{ display: 'flex', alignItems: 'center', flex: i < steps.length - 1 ? 1 : 'none' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.82rem', fontWeight: 700, fontFamily: 'Helvetica Neue, Arial, sans-serif', backgroundColor: i < current ? '#1A7A4A' : i === current ? '#0F2347' : '#E5E7EB', color: i <= current ? 'white' : '#9CA3AF' }}>
              {i < current ? '✓' : i + 1}
            </div>
            <span style={{ fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', color: i === current ? '#0F2347' : '#9CA3AF', fontWeight: i === current ? 600 : 400 }}>{s}</span>
          </div>
          {i < steps.length - 1 && <div style={{ flex: 1, height: 2, backgroundColor: i < current ? '#1A7A4A' : '#E5E7EB', margin: '0 1rem' }} />}
        </div>
      ))}
    </div>
  )
}

function FileUploadZone({ label, accept, required, hint, onChange, file, multiple }) {
  const [drag, setDrag] = useState(false)
  const inputId = `file-${label.replace(/\s+/g, '-')}`

  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    onChange(multiple ? Array.from(e.dataTransfer.files) : e.dataTransfer.files[0])
  }

  const fileLabel = file
    ? multiple
      ? `${file.length} fichier(s) sélectionné(s)`
      : file.name
    : null

  return (
    <div style={{ marginBottom: '1.25rem' }}>
      <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#4B5563', marginBottom: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
        {label} {required && <span style={{ color: '#C0392B' }}>*</span>}
      </label>
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById(inputId).click()}
        style={{ border: `2px dashed ${drag ? '#0F2347' : file ? '#1A7A4A' : '#E5E7EB'}`, borderRadius: 8, padding: '1.5rem', textAlign: 'center', cursor: 'pointer', backgroundColor: drag ? 'rgba(15,35,71,0.03)' : file ? '#F0FAF4' : '#F9FAFB', transition: 'all 0.2s' }}
      >
        <input id={inputId} type="file" accept={accept} multiple={!!multiple} style={{ display: 'none' }}
          onChange={e => onChange(multiple ? Array.from(e.target.files) : e.target.files[0])} />
        {fileLabel ? (
          <div>
            <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>✓</div>
            <div style={{ fontSize: '0.85rem', color: '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600 }}>{fileLabel}</div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem', opacity: 0.4 }}>↑</div>
            <div style={{ fontSize: '0.85rem', color: '#6B7280', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
              Glissez-déposez ou <span style={{ color: '#0F2347', fontWeight: 600 }}>parcourir</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: '#9CA3AF', marginTop: 4, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{hint}</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function NewClaim() {
  const navigate = useNavigate()
  const [dark] = useDarkMode()
  const [step, setStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [claimRef, setClaimRef] = useState('')
  const [claimId, setClaimId] = useState('')
  const [equipmentList, setEquipmentList] = useState([])
  const [equipLoading, setEquipLoading] = useState(true)

  const [form, setForm] = useState({
    equipmentId: '',
    incidentDate: '',
    description: '',
    amount: '',
  })
  const [files, setFiles] = useState({ csv: null, photos: null, pdf: null })

  useEffect(() => {
    api.get('/equipment')
      .then(res => {
        const inner = res.data?.data ?? res.data
        const arr = inner?.data ?? inner
        setEquipmentList(Array.isArray(arr) ? arr : [])
      })
      .catch(err => console.error('Equipment fetch error:', err))
      .finally(() => setEquipLoading(false))
  }, [])

  const handleNext = () => {
    setError('')
    if (step === 0) {
      if (!form.equipmentId || !form.incidentDate || !form.description || !form.amount) {
        setError('Veuillez remplir tous les champs obligatoires.'); return
      }
      if (form.description.trim().length < 20) {
        setError('La description doit contenir au moins 20 caractères.'); return
      }
      const amt = Number(form.amount)
      if (isNaN(amt) || amt <= 0) {
        setError('Le montant doit être un nombre positif.'); return
      }
      if (amt > 500_000_000) {
        setError('Le montant ne peut pas dépasser 500 000 000 DA.'); return
      }
      const date = new Date(form.incidentDate)
      if (date > new Date()) {
        setError("La date d'incident ne peut pas être dans le futur."); return
      }
    }
    if (step === 1) {
      if (!files.csv) { setError('Le fichier CSV est obligatoire.'); return }
      if (!files.photos || files.photos.length === 0) { setError('Au moins une photo est obligatoire.'); return }
    }
    setStep(s => s + 1)
  }

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('equipmentId', form.equipmentId)
      formData.append('incidentDate', form.incidentDate)
      formData.append('description', form.description.trim())
      formData.append('claimedAmount', form.amount)

      if (files.csv) formData.append('files', files.csv)
      if (files.photos) files.photos.forEach(p => formData.append('files', p))
      if (files.pdf) formData.append('files', files.pdf)

      const res = await api.post('/claims', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const data = res.data?.data ?? res.data
      setClaimRef(data.reference)
      setClaimId(data.claimId)
      setStep(3)
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la soumission.')
    } finally {
      setLoading(false)
    }
  }

  const selectedEquipment = equipmentList.find(e => e.id === form.equipmentId)

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const gold = '#C9A84C'
  const navy = '#0F2347'

  const inputStyle = {
    width: '100%', padding: '0.72rem 0.9rem',
    border: '1.5px solid #E5E7EB', borderRadius: 6,
    fontSize: '0.9rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
    outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box',
  }
  const labelStyle = {
    display: 'block', fontSize: '0.74rem', fontWeight: 600,
    textTransform: 'uppercase', letterSpacing: '0.06em',
    color: '#4B5563', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
      <Sidebar role="CLIENT" dark={dark} />

      {/* Main content — full flex column to allow vertical centering */}
      <div style={{
        marginLeft: 240, flex: 1,
        display: 'flex', flexDirection: 'column',
        minHeight: '100vh',
      }}>
        {/* Page title strip */}
        <div style={{ padding: '2rem 2rem 0' }}>
          <p style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.12em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Sinistres</p>
          <h1 style={{ fontSize: '1.75rem', color: textMain, fontWeight: 400 }}><strong>Nouveau</strong> sinistre</h1>
        </div>

        {/* Centered form area */}
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: step === 3 ? 'center' : 'flex-start',
          justifyContent: 'center',
          padding: '2rem',
        }}>
          <div style={{
            width: '100%',
            maxWidth: 680,
            backgroundColor: cardBg,
            borderRadius: 16,
            border: `1px solid ${cardBorder}`,
            padding: '2.5rem',
            boxShadow: dark
              ? '0 20px 60px rgba(0,0,0,0.35)'
              : '0 8px 40px rgba(15,35,71,0.08)',
          }}>

            {/* ── Success screen ── */}
            {step === 3 ? (
              <div style={{ textAlign: 'center', padding: '2rem 0' }}>
                <div style={{ width: 64, height: 64, borderRadius: '50%', backgroundColor: '#F0FAF4', border: '2px solid #B8E4CA', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', margin: '0 auto 1.5rem' }}>✓</div>
                <h2 style={{ color: textMain, fontSize: '1.5rem', fontWeight: 400, marginBottom: '0.5rem' }}>
                  Sinistre <strong>soumis avec succès</strong>
                </h2>
                <p style={{ color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                  Votre dossier est en cours d'analyse par notre système IA
                </p>
                <div style={{ backgroundColor: pageBg, border: `1px solid ${cardBorder}`, borderRadius: 8, padding: '1rem 2rem', display: 'inline-block', marginBottom: '2rem' }}>
                  <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.25rem' }}>Référence</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: gold, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{claimRef}</div>
                </div>
                <div style={{ backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, padding: '0.75rem', marginBottom: '2rem', fontSize: '0.85rem', color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                  Temps d'analyse estimé : moins de 5 minutes
                </div>
                <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                  <button
                    onClick={() => navigate(`/client/claims/${claimId}`)}
                    style={{ padding: '0.75rem 1.5rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
                    Voir le dossier
                  </button>
                  <button
                    onClick={() => navigate('/client/claims')}
                    style={{ padding: '0.75rem 1.5rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
                    Mes sinistres
                  </button>
                </div>
              </div>
            ) : (
              <>
                <StepIndicator current={step} />

                {error && (
                  <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 6, padding: '0.7rem 0.9rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                    ⚠ {error}
                  </div>
                )}

                {/* ── Step 0: Informations ── */}
                {step === 0 && (
                  <div>
                    <h3 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Informations du sinistre</h3>

                    <div style={{ marginBottom: '1rem' }}>
                      <label style={labelStyle}>Équipement concerné <span style={{ color: '#C0392B' }}>*</span></label>
                      {equipLoading ? (
                        <div style={{ padding: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>Chargement des équipements...</div>
                      ) : equipmentList.length === 0 ? (
                        <div style={{ padding: '0.75rem', backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 6, color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.85rem' }}>
                          Aucun équipement enregistré.{' '}
                          <span style={{ textDecoration: 'underline', cursor: 'pointer' }} onClick={() => navigate('/client/profile')}>
                            Ajoutez-en un dans votre profil.
                          </span>
                        </div>
                      ) : (
                        <select value={form.equipmentId} onChange={e => setForm({ ...form, equipmentId: e.target.value })} style={inputStyle}>
                          <option value="">Sélectionnez un équipement</option>
                          {equipmentList.map(eq => (
                            <option key={eq.id} value={eq.id}>
                              {eq.name}{eq.serialNumber ? ` — ${eq.serialNumber}` : ''}{eq.type ? ` (${eq.type})` : ''}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                      <div>
                        <label style={labelStyle}>Date de l'incident <span style={{ color: '#C0392B' }}>*</span></label>
                        <input
                          type="date"
                          value={form.incidentDate}
                          max={new Date().toISOString().split('T')[0]}
                          onChange={e => setForm({ ...form, incidentDate: e.target.value })}
                          style={inputStyle}
                        />
                      </div>
                      <div>
                        <label style={labelStyle}>Montant réclamé (DA) <span style={{ color: '#C0392B' }}>*</span></label>
                        <input
                          type="number"
                          placeholder="Ex: 850000"
                          min="1"
                          max="500000000"
                          value={form.amount}
                          onChange={e => setForm({ ...form, amount: e.target.value })}
                          style={inputStyle}
                        />
                      </div>
                    </div>

                    <div style={{ marginBottom: '1.5rem' }}>
                      <label style={labelStyle}>Description <span style={{ color: '#C0392B' }}>*</span></label>
                      <textarea
                        rows={4}
                        placeholder="Décrivez en détail ce qui s'est passé (minimum 20 caractères)..."
                        value={form.description}
                        onChange={e => setForm({ ...form, description: e.target.value })}
                        style={{ ...inputStyle, resize: 'vertical' }}
                      />
                      <div style={{ fontSize: '0.72rem', color: form.description.trim().length >= 20 ? '#1A7A4A' : '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                        {form.description.trim().length}/20 caractères minimum
                      </div>
                    </div>
                  </div>
                )}

                {/* ── Step 1: Documents ── */}
                {step === 1 && (
                  <div>
                    <h3 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, marginBottom: '0.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Documents justificatifs</h3>
                    <p style={{ color: textSub, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>
                      Ces fichiers seront analysés par nos modèles IA
                    </p>
                    <FileUploadZone
                      label="Données capteurs CSV"
                      accept=".csv,text/csv"
                      required hint="Fichier CSV avec données capteurs (min 6 mois)"
                      file={files.csv}
                      onChange={f => setFiles({ ...files, csv: f })}
                    />
                    <FileUploadZone
                      label="Photos équipement"
                      accept="image/*"
                      required multiple hint="JPG, PNG — 1 à 10 photos"
                      file={files.photos}
                      onChange={f => setFiles({ ...files, photos: f })}
                    />
                    <FileUploadZone
                      label="Rapport technique PDF"
                      accept=".pdf,application/pdf"
                      hint="Optionnel — rapport d'expertise"
                      file={files.pdf}
                      onChange={f => setFiles({ ...files, pdf: f })}
                    />
                  </div>
                )}

                {/* ── Step 2: Confirmation ── */}
                {step === 2 && (
                  <div>
                    <h3 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, marginBottom: '1.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Confirmation</h3>
                    <div style={{ backgroundColor: pageBg, borderRadius: 8, border: `1px solid ${cardBorder}`, overflow: 'hidden', marginBottom: '1.5rem' }}>
                      <div style={{ padding: '0.75rem 1rem', backgroundColor: navy }}>
                        <span style={{ color: 'white', fontSize: '0.82rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Récapitulatif</span>
                      </div>
                      {[
                        ['Équipement', selectedEquipment?.name || '—'],
                        ['Date incident', form.incidentDate ? new Date(form.incidentDate).toLocaleDateString('fr-FR') : '—'],
                        ['Montant', form.amount ? `${Number(form.amount).toLocaleString('fr-FR')} DA` : '—'],
                        ['CSV capteurs', files.csv?.name || '—'],
                        ['Photos', files.photos ? `${files.photos.length} photo(s)` : '—'],
                        ['Rapport PDF', files.pdf?.name || 'Non fourni'],
                      ].map(([k, v]) => (
                        <div key={k} style={{ display: 'flex', padding: '0.65rem 1rem', borderBottom: `1px solid ${cardBorder}` }}>
                          <span style={{ width: 160, fontSize: '0.82rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{k}</span>
                          <span style={{ fontSize: '0.85rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</span>
                        </div>
                      ))}
                    </div>
                    <div style={{ backgroundColor: '#FEF9E7', border: '1px solid #F7DC6F', borderRadius: 8, padding: '0.75rem 1rem', fontSize: '0.82rem', color: '#7D6608', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                      En soumettant, vous certifiez que toutes les informations sont exactes et véridiques.
                    </div>
                  </div>
                )}

                {/* ── Nav buttons ── */}
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1.5rem', paddingTop: '1.5rem', borderTop: `1px solid ${cardBorder}` }}>
                  <button
                    onClick={() => step === 0 ? navigate('/client/dashboard') : setStep(s => s - 1)}
                    style={{ padding: '0.75rem 1.5rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
                    {step === 0 ? 'Annuler' : '← Précédent'}
                  </button>

                  {step < 2 ? (
                    <button onClick={handleNext}
                      style={{ padding: '0.75rem 1.5rem', background: `linear-gradient(135deg, ${navy}, #1A3A6B)`, color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', fontWeight: 600 }}>
                      Suivant →
                    </button>
                  ) : (
                    <button onClick={handleSubmit} disabled={loading}
                      style={{ padding: '0.75rem 1.5rem', background: loading ? '#9CA3AF' : 'linear-gradient(135deg, #1A7A4A, #27AE60)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 600 }}>
                      {loading ? 'Soumission...' : '✓ Soumettre le sinistre'}
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}