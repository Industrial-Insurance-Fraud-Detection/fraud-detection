import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import api from '../../api/axios'
import AuthLeftPanel from './AuthLeftPanel'

// ── Password strength checker (same as AuthPage) ──────────────────────────────
function PasswordStrength({ password }) {
    if (!password) return null
    const checks = {
        len: password.length >= 8,
        upper: /[A-Z]/.test(password),
        lower: /[a-z]/.test(password),
        digit: /[0-9]/.test(password),
    }
    const passed = Object.values(checks).filter(Boolean).length
    const bar = ['#E5E7EB', '#C0392B', '#E67E22', '#F39C12', '#1A7A4A']
    const lbl = ['', 'Très faible', 'Faible', 'Moyen', 'Fort']
    return (
        <div style={{ marginTop: '0.55rem' }}>
            <div style={{ display: 'flex', gap: 4, marginBottom: '0.45rem' }}>
                {[1, 2, 3, 4].map(i => (
                    <div key={i} style={{ flex: 1, height: 4, borderRadius: 3, backgroundColor: i <= passed ? bar[passed] : '#F3F4F6', transition: 'background-color 0.35s' }} />
                ))}
            </div>
            <div style={{ fontSize: '0.7rem', color: bar[passed], fontFamily: 'Helvetica Neue,Arial,sans-serif', fontWeight: 700, marginBottom: '0.45rem' }}>
                {lbl[passed]}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 1rem' }}>
                {[
                    [checks.len, '8 caractères min.'],
                    [checks.upper, 'Majuscule (A–Z)'],
                    [checks.lower, 'Minuscule (a–z)'],
                    [checks.digit, 'Chiffre (0–9)'],
                ].map(([ok, txt]) => (
                    <div key={txt} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.68rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', color: ok ? '#1A7A4A' : '#9CA3AF', transition: 'color 0.2s' }}>
                        <span style={{ fontWeight: 700, fontSize: '0.7rem' }}>{ok ? '✓' : '○'}</span>
                        {txt}
                    </div>
                ))}
            </div>
        </div>
    )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function ResetPasswordPage() {
    const navigate = useNavigate()
    const location = useLocation()

    const [token, setToken] = useState(location.state?.token || '')
    const [newPassword, setNewPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState(false)
    const [showPass, setShowPass] = useState(false)

    const inp = {
        width: '100%', padding: '0.75rem 1rem',
        border: '1.5px solid #E5E7EB', borderRadius: 8,
        fontSize: '0.88rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
        outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box',
        transition: 'border-color 0.2s, box-shadow 0.2s', color: '#0F2347',
    }
    const lbl = {
        display: 'block', fontSize: '0.7rem', fontWeight: 600,
        textTransform: 'uppercase', letterSpacing: '0.08em',
        color: '#6B7280', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
    }
    const onFocus = e => { e.target.style.borderColor = '#C9A84C'; e.target.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.12)' }
    const onBlur = e => { e.target.style.borderColor = '#E5E7EB'; e.target.style.boxShadow = 'none' }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        if (!token.trim()) { setError('Le jeton de réinitialisation est requis.'); return }
        if (newPassword.length < 8) { setError('Le mot de passe doit contenir au moins 8 caractères.'); return }
        if (!/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/.test(newPassword)) {
            setError('Le mot de passe doit contenir une majuscule, une minuscule et un chiffre.'); return
        }
        if (newPassword !== confirmPassword) { setError('Les mots de passe ne correspondent pas.'); return }
        setLoading(true)
        try {
            await api.post('/auth/reset-password', { token: token.trim(), newPassword })
            setSuccess(true)
        } catch (err) {
            const msg = err.response?.data?.message
            setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Jeton invalide ou expiré.')
        } finally { setLoading(false) }
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia,serif' }}>

            <AuthLeftPanel />

            {/* RIGHT */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '2.5rem 4rem', backgroundColor: 'white', overflowY: 'auto' }}>
                <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

                    {/* Back link */}
                    {!success && (
                        <button onClick={() => navigate('/forgot-password')} style={{
                            background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.8rem',
                            cursor: 'pointer', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                            padding: 0, marginBottom: '1.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem',
                        }}>
                            ← Retour
                        </button>
                    )}

                    {success ? (
                        /* ── Success state ── */
                        <div style={{ textAlign: 'center', paddingTop: '1rem' }}>
                            <div style={{
                                width: 72, height: 72, borderRadius: '50%',
                                backgroundColor: '#F0FAF4', border: '2px solid #B8E4CA',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '2rem', margin: '0 auto 1.5rem',
                            }}>✓</div>
                            <h2 style={{ color: '#0F2347', fontSize: '1.75rem', fontWeight: 400, marginBottom: '0.5rem' }}>
                                Mot de passe <strong>réinitialisé !</strong>
                            </h2>
                            <p style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue,Arial,sans-serif', fontSize: '0.85rem', marginBottom: '2rem', lineHeight: 1.6 }}>
                                Votre mot de passe a été modifié avec succès.<br />Vous pouvez maintenant vous connecter.
                            </p>
                            <button onClick={() => navigate('/login')} style={{
                                width: '100%', padding: '0.82rem',
                                background: 'linear-gradient(135deg, #0F2347 0%, #1A3A6B 100%)',
                                color: 'white', border: 'none', borderRadius: 8,
                                fontSize: '0.86rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                                fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
                                cursor: 'pointer', boxShadow: '0 4px 18px rgba(15,35,71,0.28)',
                            }}>
                                Se connecter →
                            </button>
                        </div>
                    ) : (
                        /* ── Form ── */
                        <>
                            <p style={{ fontSize: '0.68rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.35rem', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
                                Sécurité
                            </p>
                            <h2 style={{ color: '#0F2347', fontSize: '1.85rem', fontWeight: 400, marginBottom: '0.35rem', letterSpacing: '-0.02em' }}>
                                <strong>Réinitialiser</strong> le mot de passe
                            </h2>
                            <p style={{ color: '#9CA3AF', fontSize: '0.83rem', marginBottom: '1.75rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', lineHeight: 1.55 }}>
                                Saisissez le jeton reçu et choisissez votre nouveau mot de passe.
                            </p>

                            {error && (
                                <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.65rem 0.9rem', color: '#C0392B', fontSize: '0.81rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginBottom: '1.25rem', display: 'flex', gap: '0.5rem' }}>
                                    <span>⚠</span> {error}
                                </div>
                            )}

                            <form onSubmit={handleSubmit}>
                                {/* Token field */}
                                <div style={{ marginBottom: '1.1rem' }}>
                                    <label style={lbl}>Jeton de réinitialisation <span style={{ color: '#C0392B' }}>*</span></label>
                                    <input type="text" placeholder="Collez le jeton ici…" value={token}
                                        onChange={e => setToken(e.target.value)}
                                        style={{ ...inp, fontFamily: 'monospace', fontSize: '0.82rem', letterSpacing: '0.02em' }}
                                        onFocus={onFocus} onBlur={onBlur} />
                                </div>

                                {/* New password */}
                                <div style={{ marginBottom: '1rem' }}>
                                    <label style={lbl}>Nouveau mot de passe <span style={{ color: '#C0392B' }}>*</span></label>
                                    <div style={{ position: 'relative' }}>
                                        <input type={showPass ? 'text' : 'password'} placeholder="Minimum 8 caractères"
                                            value={newPassword} onChange={e => setNewPassword(e.target.value)}
                                            style={{ ...inp, paddingRight: '2.8rem' }} onFocus={onFocus} onBlur={onBlur} />
                                        <button type="button" onClick={() => setShowPass(!showPass)} style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF', padding: 0, fontSize: '1rem' }}>
                                            {showPass ? '🙈' : '👁'}
                                        </button>
                                    </div>
                                    <PasswordStrength password={newPassword} />
                                </div>

                                {/* Confirm password */}
                                <div style={{ marginBottom: '1.5rem' }}>
                                    <label style={lbl}>Confirmer le mot de passe <span style={{ color: '#C0392B' }}>*</span></label>
                                    <div style={{ position: 'relative' }}>
                                        <input type={showPass ? 'text' : 'password'} placeholder="Répétez le nouveau mot de passe"
                                            value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)}
                                            style={{
                                                ...inp, paddingRight: '2.8rem',
                                                borderColor: confirmPassword && confirmPassword !== newPassword ? '#C0392B' : '#E5E7EB',
                                            }}
                                            onFocus={onFocus} onBlur={onBlur} />
                                    </div>
                                    {confirmPassword && confirmPassword !== newPassword && (
                                        <div style={{ fontSize: '0.68rem', color: '#C0392B', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginTop: '0.25rem' }}>✗ Les mots de passe ne correspondent pas</div>
                                    )}
                                    {confirmPassword && confirmPassword === newPassword && (
                                        <div style={{ fontSize: '0.68rem', color: '#1A7A4A', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginTop: '0.25rem' }}>✓ Les mots de passe correspondent</div>
                                    )}
                                </div>

                                <button type="submit" disabled={loading} style={{
                                    width: '100%', padding: '0.82rem',
                                    background: loading ? '#9CA3AF' : 'linear-gradient(135deg, #0F2347 0%, #1A3A6B 100%)',
                                    color: 'white', border: 'none', borderRadius: 8,
                                    fontSize: '0.86rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                                    fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
                                    cursor: loading ? 'not-allowed' : 'pointer',
                                    boxShadow: loading ? 'none' : '0 4px 18px rgba(15,35,71,0.28)',
                                }}>
                                    {loading ? 'Réinitialisation…' : 'Réinitialiser le mot de passe →'}
                                </button>
                            </form>

                            <p style={{ textAlign: 'center', marginTop: '1.1rem', fontSize: '0.79rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
                                Retour à la{' '}
                                <span onClick={() => navigate('/login')} style={{ color: '#C9A84C', fontWeight: 600, cursor: 'pointer' }}>
                                    connexion
                                </span>
                            </p>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}