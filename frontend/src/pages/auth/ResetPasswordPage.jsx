import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import api from '../../api/axios'

/**
 * ResetPasswordPage
 * POST /auth/reset-password  →  { message }
 * Accepts token from: URL state (passed from ForgotPasswordPage) or manual entry.
 */
export default function ResetPasswordPage() {
    const navigate = useNavigate()
    const location = useLocation()

    const [token, setToken] = useState(location.state?.token || '')
    const [newPassword, setNewPassword] = useState('')
    const [confirmPassword, setConfirmPassword] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [success, setSuccess] = useState(false)
    const [showPassword, setShowPassword] = useState(false)

    const passwordStrength = () => {
        if (newPassword.length === 0) return { level: 0, label: '', color: '#E5E7EB' }
        if (newPassword.length < 6) return { level: 1, label: 'Faible', color: '#C0392B' }
        if (newPassword.length < 10) return { level: 2, label: 'Moyen', color: '#F39C12' }
        if (/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/.test(newPassword)) return { level: 3, label: 'Fort', color: '#1A7A4A' }
        return { level: 2, label: 'Moyen', color: '#F39C12' }
    }

    const strength = passwordStrength()

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        if (!token.trim()) { setError('Veuillez saisir le jeton de réinitialisation.'); return }
        if (newPassword.length < 8) { setError('Le mot de passe doit contenir au moins 8 caractères.'); return }
        if (!/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/.test(newPassword)) {
            setError('Le mot de passe doit contenir au moins une majuscule, une minuscule et un chiffre.')
            return
        }
        if (newPassword !== confirmPassword) { setError('Les mots de passe ne correspondent pas.'); return }

        setLoading(true)
        try {
            await api.post('/auth/reset-password', { token: token.trim(), newPassword })
            setSuccess(true)
        } catch (err) {
            const msg = err.response?.data?.message
            setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Jeton invalide ou expiré.')
        } finally {
            setLoading(false)
        }
    }

    const inputStyle = {
        width: '100%', padding: '0.75rem 1rem',
        border: '1.5px solid #E5E7EB', borderRadius: 8,
        fontSize: '0.92rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
        outline: 'none', backgroundColor: '#F9FAFB',
        boxSizing: 'border-box', transition: 'border-color 0.2s',
    }
    const labelStyle = {
        display: 'block', fontSize: '0.74rem', fontWeight: 600,
        textTransform: 'uppercase', letterSpacing: '0.06em',
        color: '#4B5563', marginBottom: '0.4rem',
        fontFamily: 'Helvetica Neue, Arial, sans-serif',
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia, serif', backgroundColor: '#F7F8FC' }}>

            {/* Left panel */}
            <div style={{
                width: '45%',
                background: 'linear-gradient(155deg, #0F2347 0%, #1A3A6B 60%, #0D1E3D 100%)',
                display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
                padding: '3rem', position: 'relative', overflow: 'hidden',
            }}>
                <div style={{ position: 'absolute', top: 0, right: 0, width: 280, height: 280, borderRadius: '50%', background: 'radial-gradient(circle, #C9A84C, transparent)', opacity: 0.12, transform: 'translate(30%, -30%)' }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', zIndex: 1 }}>
                    <div style={{ width: 42, height: 42, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🛡</div>
                    <div>
                        <div style={{ color: 'white', fontWeight: 700, fontSize: '1.1rem' }}>FraudGuard AI</div>
                        <div style={{ color: '#C9A84C', fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase' }}>Industrial Insurance</div>
                    </div>
                </div>
                <div style={{ zIndex: 1 }}>
                    <div style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>🔑</div>
                    <h1 style={{ color: 'white', fontSize: '2rem', fontWeight: 400, lineHeight: 1.3, marginBottom: '1rem' }}>
                        Nouveau<br />
                        <span style={{ color: '#C9A84C', fontStyle: 'italic' }}>mot de passe</span>
                    </h1>
                    <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.9rem', lineHeight: 1.7, fontFamily: 'Helvetica Neue, Arial, sans-serif', maxWidth: 300 }}>
                        Choisissez un mot de passe fort contenant au moins 8 caractères avec majuscule, minuscule et chiffre.
                    </p>
                    <div style={{ marginTop: '2rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {['Au moins 8 caractères', 'Une lettre majuscule', 'Une lettre minuscule', 'Un chiffre'].map(r => (
                            <div key={r} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#C9A84C', flexShrink: 0 }} />
                                <span style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{r}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <div style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.72rem', zIndex: 1, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    2026 FraudGuard AI
                </div>
            </div>

            {/* Right panel */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '3rem 4rem', backgroundColor: 'white' }}>
                <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

                    <button onClick={() => navigate('/forgot-password')}
                        style={{ background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        ← Retour
                    </button>

                    <p style={{ fontSize: '0.72rem', letterSpacing: '0.15em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                        Sécurité
                    </p>
                    <h2 style={{ color: '#0F2347', fontSize: '2rem', fontWeight: 400, marginBottom: '0.5rem' }}>
                        <strong>Réinitialiser</strong> le mot de passe
                    </h2>
                    <p style={{ color: '#9CA3AF', fontSize: '0.86rem', marginBottom: '2rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.5 }}>
                        Saisissez le jeton reçu et choisissez votre nouveau mot de passe.
                    </p>

                    {error && (
                        <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.75rem 1rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>
                            ⚠ {error}
                        </div>
                    )}

                    {success ? (
                        <div>
                            <div style={{ textAlign: 'center', padding: '2rem 0' }}>
                                <div style={{ width: 64, height: 64, borderRadius: '50%', backgroundColor: '#F0FAF4', border: '2px solid #B8E4CA', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.75rem', margin: '0 auto 1.5rem' }}>✓</div>
                                <h3 style={{ color: '#0F2347', fontSize: '1.4rem', fontWeight: 400, marginBottom: '0.5rem' }}>
                                    Mot de passe <strong>réinitialisé !</strong>
                                </h3>
                                <p style={{ color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.88rem', marginBottom: '2rem', lineHeight: 1.5 }}>
                                    Votre mot de passe a été modifié avec succès. Vous pouvez maintenant vous connecter avec votre nouveau mot de passe.
                                </p>
                                <button onClick={() => navigate('/login')}
                                    style={{ width: '100%', padding: '0.85rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                                    Se connecter
                                </button>
                            </div>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit}>
                            {/* Token */}
                            <div style={{ marginBottom: '1.25rem' }}>
                                <label style={labelStyle}>Jeton de réinitialisation <span style={{ color: '#C0392B' }}>*</span></label>
                                <input
                                    type="text"
                                    placeholder="Collez le jeton reçu ici..."
                                    value={token}
                                    onChange={e => setToken(e.target.value)}
                                    style={{ ...inputStyle, fontFamily: 'monospace', fontSize: '0.82rem' }}
                                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                                    onBlur={e => e.target.style.borderColor = '#E5E7EB'}
                                />
                            </div>

                            {/* New password */}
                            <div style={{ marginBottom: '1rem' }}>
                                <label style={labelStyle}>Nouveau mot de passe <span style={{ color: '#C0392B' }}>*</span></label>
                                <div style={{ position: 'relative' }}>
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        placeholder="Minimum 8 caractères"
                                        value={newPassword}
                                        onChange={e => setNewPassword(e.target.value)}
                                        style={{ ...inputStyle, paddingRight: '3rem' }}
                                        onFocus={e => e.target.style.borderColor = '#0F2347'}
                                        onBlur={e => e.target.style.borderColor = '#E5E7EB'}
                                    />
                                    <button type="button" onClick={() => setShowPassword(!showPassword)}
                                        style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF', fontSize: '0.9rem' }}>
                                        {showPassword ? '🙈' : '👁'}
                                    </button>
                                </div>

                                {/* Strength bar */}
                                {newPassword.length > 0 && (
                                    <div style={{ marginTop: '0.5rem' }}>
                                        <div style={{ height: 5, backgroundColor: '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                                            <div style={{ height: '100%', borderRadius: 3, transition: 'width 0.3s, background 0.3s', width: `${(strength.level / 3) * 100}%`, backgroundColor: strength.color }} />
                                        </div>
                                        <div style={{ fontSize: '0.72rem', color: strength.color, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.2rem', fontWeight: 600 }}>
                                            {strength.label}
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Confirm password */}
                            <div style={{ marginBottom: '1.5rem' }}>
                                <label style={labelStyle}>Confirmer le mot de passe <span style={{ color: '#C0392B' }}>*</span></label>
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    placeholder="Répétez le nouveau mot de passe"
                                    value={confirmPassword}
                                    onChange={e => setConfirmPassword(e.target.value)}
                                    style={{ ...inputStyle, borderColor: confirmPassword && confirmPassword !== newPassword ? '#C0392B' : '#E5E7EB' }}
                                    onFocus={e => e.target.style.borderColor = confirmPassword && confirmPassword !== newPassword ? '#C0392B' : '#0F2347'}
                                    onBlur={e => e.target.style.borderColor = confirmPassword && confirmPassword !== newPassword ? '#C0392B' : '#E5E7EB'}
                                />
                                {confirmPassword && confirmPassword !== newPassword && (
                                    <div style={{ fontSize: '0.72rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                                        Les mots de passe ne correspondent pas
                                    </div>
                                )}
                                {confirmPassword && confirmPassword === newPassword && (
                                    <div style={{ fontSize: '0.72rem', color: '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.25rem' }}>
                                        ✓ Les mots de passe correspondent
                                    </div>
                                )}
                            </div>

                            <button type="submit" disabled={loading}
                                style={{ width: '100%', padding: '0.85rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 15px rgba(15,35,71,0.25)' }}>
                                {loading ? 'Réinitialisation...' : 'Réinitialiser le mot de passe'}
                            </button>
                        </form>
                    )}

                    <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                        Retour à la{' '}
                        <span onClick={() => navigate('/login')}
                            style={{ color: '#0F2347', fontWeight: 600, cursor: 'pointer', textDecoration: 'underline' }}>
                            connexion
                        </span>
                    </p>
                </div>
            </div>
        </div>
    )
}