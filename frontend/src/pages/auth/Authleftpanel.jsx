import { useEffect, useRef } from 'react'

// ── Shared Left Panel — used across all 3 auth pages ─────────────────────────
export default function AuthLeftPanel() {
    const mountRef = useRef(null)
    const sceneRef = useRef(null)

    useEffect(() => {
        // Dynamically import Three.js to keep the bundle light
        let animFrameId
        let renderer

        const init = async () => {
            const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js')
                .catch(() => null)

            if (!THREE || !mountRef.current) return

            const W = mountRef.current.clientWidth
            const H = mountRef.current.clientHeight

            // ── Renderer ────────────────────────────────────────────────────
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
            renderer.setSize(W, H)
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
            renderer.setClearColor(0x000000, 0)
            mountRef.current.appendChild(renderer.domElement)

            // ── Scene / Camera ───────────────────────────────────────────────
            const scene = new THREE.Scene()
            const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100)
            camera.position.set(0, 1.2, 5.5)
            camera.lookAt(0, 0, 0)

            // ── Lights ───────────────────────────────────────────────────────
            const ambientLight = new THREE.AmbientLight(0x1a2a45, 0.8)
            scene.add(ambientLight)

            // Warm gold from upper-left
            const warmLight = new THREE.DirectionalLight(0xE8C97A, 2.2)
            warmLight.position.set(-4, 4, 2)
            scene.add(warmLight)

            // Cold blue from lower-right
            const coldLight = new THREE.DirectionalLight(0x4488CC, 1.4)
            coldLight.position.set(4, -2, -2)
            scene.add(coldLight)

            const fillLight = new THREE.PointLight(0xC9A84C, 0.4, 10)
            fillLight.position.set(0, 0, 3)
            scene.add(fillLight)

            // ── Dark metallic material ───────────────────────────────────────
            const metalMat = new THREE.MeshStandardMaterial({
                color: 0x1A2A45,
                roughness: 0.6,
                metalness: 0.85,
            })
            const darkMat = new THREE.MeshStandardMaterial({
                color: 0x0F1A2E,
                roughness: 0.7,
                metalness: 0.9,
            })
            const goldMat = new THREE.MeshStandardMaterial({
                color: 0xC9A84C,
                roughness: 0.3,
                metalness: 0.95,
            })

            // ── Procedural Centrifugal Pump ──────────────────────────────────
            const pumpGroup = new THREE.Group()

            // Main volute casing (snail shell shape — approximated with torus + cylinder)
            const voluteCasing = new THREE.Mesh(
                new THREE.TorusGeometry(0.72, 0.38, 16, 48),
                metalMat
            )
            voluteCasing.rotation.x = Math.PI / 2
            voluteCasing.position.set(0, 0, 0)
            pumpGroup.add(voluteCasing)

            // Volute outer shell top half
            const shellTop = new THREE.Mesh(
                new THREE.CylinderGeometry(1.02, 1.02, 0.28, 40),
                metalMat
            )
            shellTop.position.set(0, 0, 0)
            pumpGroup.add(shellTop)

            // Discharge nozzle (top)
            const dischargeNozzle = new THREE.Mesh(
                new THREE.CylinderGeometry(0.22, 0.22, 0.85, 16),
                darkMat
            )
            dischargeNozzle.position.set(0, 0.9, 0)
            pumpGroup.add(dischargeNozzle)

            // Nozzle flange
            const dischargeFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(0.32, 0.32, 0.08, 16),
                goldMat
            )
            dischargeFlange.position.set(0, 1.28, 0)
            pumpGroup.add(dischargeFlange)

            // Suction nozzle (left, horizontal)
            const suctionNozzle = new THREE.Mesh(
                new THREE.CylinderGeometry(0.26, 0.26, 0.9, 16),
                darkMat
            )
            suctionNozzle.rotation.z = Math.PI / 2
            suctionNozzle.position.set(-1.0, 0, 0)
            pumpGroup.add(suctionNozzle)

            const suctionFlange = new THREE.Mesh(
                new THREE.CylinderGeometry(0.36, 0.36, 0.08, 16),
                goldMat
            )
            suctionFlange.rotation.z = Math.PI / 2
            suctionFlange.position.set(-1.42, 0, 0)
            pumpGroup.add(suctionFlange)

            // Motor coupling (right side)
            const coupling = new THREE.Mesh(
                new THREE.CylinderGeometry(0.18, 0.18, 0.28, 12),
                darkMat
            )
            coupling.rotation.z = Math.PI / 2
            coupling.position.set(1.18, 0, 0)
            pumpGroup.add(coupling)

            // Bearing housing
            const bearingHousing = new THREE.Mesh(
                new THREE.CylinderGeometry(0.30, 0.30, 0.52, 20),
                metalMat
            )
            bearingHousing.rotation.z = Math.PI / 2
            bearingHousing.position.set(1.65, 0, 0)
            pumpGroup.add(bearingHousing)

            // Motor block
            const motorBlock = new THREE.Mesh(
                new THREE.BoxGeometry(0.95, 0.72, 0.72),
                metalMat
            )
            motorBlock.position.set(2.35, 0, 0)
            pumpGroup.add(motorBlock)

            // Motor fins
            for (let i = -2; i <= 2; i++) {
                const fin = new THREE.Mesh(
                    new THREE.BoxGeometry(0.72, 0.06, 0.82),
                    darkMat
                )
                fin.position.set(2.35, i * 0.12, 0)
                pumpGroup.add(fin)
            }

            // Base plate
            const basePlate = new THREE.Mesh(
                new THREE.BoxGeometry(3.6, 0.12, 0.88),
                darkMat
            )
            basePlate.position.set(0.6, -0.52, 0)
            pumpGroup.add(basePlate)

            // Base feet
            for (const xPos of [-1.5, 1.8]) {
                const foot = new THREE.Mesh(
                    new THREE.BoxGeometry(0.28, 0.18, 0.88),
                    goldMat
                )
                foot.position.set(xPos, -0.65, 0)
                pumpGroup.add(foot)
            }

            // Shaft seal area
            const sealHousing = new THREE.Mesh(
                new THREE.CylinderGeometry(0.22, 0.26, 0.22, 16),
                darkMat
            )
            sealHousing.rotation.z = Math.PI / 2
            sealHousing.position.set(0.95, 0, 0)
            pumpGroup.add(sealHousing)

            // Scale and position the whole pump
            pumpGroup.scale.setScalar(0.82)
            pumpGroup.position.set(-0.3, -0.1, 0)
            scene.add(pumpGroup)

            // ── Background atmosphere torus ──────────────────────────────────
            const atmosphereTorus = new THREE.Mesh(
                new THREE.TorusGeometry(2.8, 0.015, 6, 120),
                new THREE.MeshBasicMaterial({ color: 0xC9A84C, transparent: true, opacity: 0.08 })
            )
            atmosphereTorus.rotation.x = Math.PI / 6
            scene.add(atmosphereTorus)

            const atmosphereTorus2 = new THREE.Mesh(
                new THREE.TorusGeometry(2.2, 0.01, 6, 100),
                new THREE.MeshBasicMaterial({ color: 0xC9A84C, transparent: true, opacity: 0.05 })
            )
            atmosphereTorus2.rotation.x = -Math.PI / 4
            atmosphereTorus2.rotation.y = Math.PI / 5
            scene.add(atmosphereTorus2)

            // ── Sensor hotspot spheres ───────────────────────────────────────
            const SENSOR_POINTS = [
                // [position, color, label, value, phase offset]
                { pos: [1.15 * 0.82 - 0.3, 0, 0.18], color: 0xC0392B, label: 'Temp: 87°C', phase: 0 },       // bearing housing — red anomaly
                { pos: [0.78 * 0.82 - 0.3, 0.88 * 0.82, 0], color: 0xC9A84C, label: 'Vibr: 4.2 mm/s', phase: 1.2 }, // discharge — amber warning
                { pos: [-1.05 * 0.82 - 0.3, 0, 0.2], color: 0x1A7A4A, label: 'Pres: 6.2 bar', phase: 2.4 },  // suction — green normal
                { pos: [0.95 * 0.82 - 0.3, 0, 0.22], color: 0x1A7A4A, label: 'Flow: 48 m³/h', phase: 3.6 },  // seal — green normal
                { pos: [2.35 * 0.82 - 0.3, 0.4 * 0.82, 0.36], color: 0xC9A84C, label: 'Score: 67/100', phase: 0.8 }, // motor — amber
            ]

            const hotspots = []
            SENSOR_POINTS.forEach(({ pos, color }) => {
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(0.055, 12, 12),
                    new THREE.MeshBasicMaterial({ color })
                )
                sphere.position.set(...pos)

                // Inner glow sphere
                const glow = new THREE.Mesh(
                    new THREE.SphereGeometry(0.1, 12, 12),
                    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18 })
                )
                glow.position.set(...pos)

                scene.add(sphere)
                scene.add(glow)
                hotspots.push({ sphere, glow, basePos: [...pos] })
            })

            // ── Grid floor ───────────────────────────────────────────────────
            const gridHelper = new THREE.GridHelper(12, 24, 0xC9A84C, 0xC9A84C)
            gridHelper.material.transparent = true
            gridHelper.material.opacity = 0.04
            gridHelper.position.y = -0.95
            scene.add(gridHelper)

            // ── Thin connector lines from hotspot to off-screen ──────────────
            SENSOR_POINTS.forEach(({ pos }, i) => {
                const endPos = [
                    pos[0] + (i % 2 === 0 ? 1.2 : -1.2),
                    pos[1] + 0.6,
                    pos[2] + 0.2
                ]
                const points = [
                    new THREE.Vector3(...pos),
                    new THREE.Vector3(...endPos)
                ]
                const lineGeo = new THREE.BufferGeometry().setFromPoints(points)
                const lineMat = new THREE.LineBasicMaterial({
                    color: 0xC9A84C,
                    transparent: true,
                    opacity: 0.25
                })
                const line = new THREE.Line(lineGeo, lineMat)
                scene.add(line)
            })

            sceneRef.current = { scene, camera, renderer, pumpGroup, hotspots, atmosphereTorus, atmosphereTorus2 }

            // ── Resize handler ───────────────────────────────────────────────
            const handleResize = () => {
                if (!mountRef.current) return
                const w = mountRef.current.clientWidth
                const h = mountRef.current.clientHeight
                camera.aspect = w / h
                camera.updateProjectionMatrix()
                renderer.setSize(w, h)
            }
            window.addEventListener('resize', handleResize)

            // ── Animation loop ───────────────────────────────────────────────
            let t = 0
            const animate = () => {
                animFrameId = requestAnimationFrame(animate)
                t += 0.008

                // Slow Y-axis rotation — one revolution every 25s → ~0.0025 rad/frame at 60fps
                pumpGroup.rotation.y = t * 0.016

                // Atmosphere rings counter-rotate
                atmosphereTorus.rotation.z = t * 0.009
                atmosphereTorus2.rotation.z = -t * 0.006

                // Hotspot pulse
                hotspots.forEach(({ glow }, i) => {
                    const phase = SENSOR_POINTS[i].phase
                    const pulse = 0.12 + 0.08 * Math.sin(t * 1.8 + phase)
                    glow.scale.setScalar(1 + pulse * 1.4)
                    glow.material.opacity = 0.12 + pulse * 0.6
                })

                renderer.render(scene, camera)
            }
            animate()
        }

        init()

        return () => {
            if (animFrameId) cancelAnimationFrame(animFrameId)
            if (renderer) {
                renderer.dispose()
                if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
                    mountRef.current.removeChild(renderer.domElement)
                }
            }
        }
    }, [])

    return (
        <div style={{
            width: '55%',
            background: 'linear-gradient(160deg, #060D1A 0%, #0A1628 45%, #0F2347 100%)',
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
            overflow: 'hidden',
            minHeight: '100vh',
        }}>
            <style>{`
                @keyframes fadeGrid  { 0%,100%{opacity:0.03} 50%{opacity:0.07} }
                @keyframes scanDown  { 0%{top:-60px} 100%{top:110%} }
                @keyframes slideUp   { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
                @keyframes particleFloat { 0%{transform:translateY(0) translateX(0);opacity:0} 10%{opacity:1} 90%{opacity:0.6} 100%{transform:translateY(-120px) translateX(20px);opacity:0} }
                @keyframes chipFloat { 0%,100%{transform:translateY(0px)} 50%{transform:translateY(-5px)} }
            `}</style>

            {/* Grid background */}
            <div style={{
                position: 'absolute', inset: 0, pointerEvents: 'none',
                backgroundImage: `
                    linear-gradient(rgba(201,168,76,0.05) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(201,168,76,0.05) 1px, transparent 1px)
                `,
                backgroundSize: '48px 48px',
                animation: 'fadeGrid 6s ease-in-out infinite',
            }} />

            {/* Scan line */}
            <div style={{
                position: 'absolute', left: 0, right: 0, height: 60,
                background: 'linear-gradient(transparent, rgba(201,168,76,0.04), transparent)',
                animation: 'scanDown 8s linear infinite',
                pointerEvents: 'none', zIndex: 1,
            }} />

            {/* Corner brackets */}
            {[
                { top: 16, left: 16, borderTop: '2px solid rgba(201,168,76,0.5)', borderLeft: '2px solid rgba(201,168,76,0.5)' },
                { top: 16, right: 16, borderTop: '2px solid rgba(201,168,76,0.5)', borderRight: '2px solid rgba(201,168,76,0.5)' },
                { bottom: 16, left: 16, borderBottom: '2px solid rgba(201,168,76,0.5)', borderLeft: '2px solid rgba(201,168,76,0.5)' },
                { bottom: 16, right: 16, borderBottom: '2px solid rgba(201,168,76,0.5)', borderRight: '2px solid rgba(201,168,76,0.5)' },
            ].map((s, i) => (
                <div key={i} style={{ position: 'absolute', width: 22, height: 22, ...s, pointerEvents: 'none', zIndex: 3 }} />
            ))}

            {/* ── Brand lockup — upper-left, small, confident ── */}
            <div style={{
                position: 'absolute', top: 28, left: 32, zIndex: 10,
                display: 'flex', alignItems: 'center', gap: '0.65rem',
                animation: 'slideUp 0.6s ease 0.1s both',
            }}>
                <div style={{
                    width: 32, height: 32, borderRadius: 7,
                    background: 'linear-gradient(135deg, #C9A84C, #E8C97A)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.95rem', boxShadow: '0 0 16px rgba(201,168,76,0.35)',
                }}>🛡</div>
                <div>
                    <div style={{ color: 'white', fontWeight: 700, fontSize: '0.88rem', letterSpacing: '0.04em', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>FraudGuard AI</div>
                    <div style={{ color: '#C9A84C', fontSize: '0.52rem', letterSpacing: '0.22em', textTransform: 'uppercase', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Industrial Insurance</div>
                </div>
            </div>

            {/* ── Three.js canvas mount ── */}
            <div
                ref={mountRef}
                style={{
                    position: 'absolute',
                    inset: 0,
                    zIndex: 2,
                }}
            />

            {/* ── Floating data chips — positioned absolutely over the scene ── */}
            {[
                { top: '24%', left: '6%', label: 'Vibration', val: '4.2 mm/s', color: '#C9A84C', phase: '0s', dur: '7s' },
                { top: '20%', right: '6%', label: 'Temp', val: '87°C', color: '#C0392B', phase: '1.5s', dur: '6s' },
                { top: '55%', left: '4%', label: 'Pression', val: '6.2 bar', color: '#1A7A4A', phase: '3s', dur: '8s' },
                { top: '60%', right: '4%', label: 'Débit', val: '48 m³/h', color: '#1A7A4A', phase: '2s', dur: '7.5s' },
                { top: '38%', right: '5%', label: 'Score', val: '67/100', color: '#C9A84C', phase: '4s', dur: '6.5s' },
            ].map((chip, i) => (
                <div key={i} style={{
                    position: 'absolute',
                    top: chip.top, left: chip.left, right: chip.right,
                    zIndex: 5,
                    background: 'rgba(6,13,26,0.85)',
                    border: `1px solid rgba(201,168,76,0.22)`,
                    borderLeft: `2px solid ${chip.color}`,
                    borderRadius: 8,
                    padding: '0.4rem 0.75rem',
                    backdropFilter: 'blur(12px)',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.45)',
                    animation: `chipFloat ${chip.dur} ease-in-out infinite`,
                    animationDelay: chip.phase,
                }}>
                    <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.58rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.15rem' }}>{chip.label}</div>
                    <div style={{ color: chip.color, fontWeight: 700, fontSize: '0.88rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{chip.val}</div>
                </div>
            ))}

            {/* ── Floating particles ── */}
            {[
                { left: '18%', bottom: '22%', size: 3, delay: '0s', dur: '4s' },
                { left: '72%', bottom: '18%', size: 2, delay: '1.5s', dur: '5s' },
                { left: '42%', bottom: '14%', size: 4, delay: '2.8s', dur: '3.5s' },
                { left: '58%', bottom: '28%', size: 2, delay: '4s', dur: '6s' },
            ].map((p, i) => (
                <div key={i} style={{
                    position: 'absolute', left: p.left, bottom: p.bottom,
                    width: p.size, height: p.size, borderRadius: '50%',
                    background: '#C9A84C', opacity: 0, zIndex: 4,
                    animation: `particleFloat ${p.dur} ease-in-out infinite`,
                    animationDelay: p.delay,
                }} />
            ))}

            {/* ── Bottom text block — lower third, left-aligned ── */}
            <div style={{
                position: 'absolute',
                bottom: 92,
                left: 0,
                right: 0,
                padding: '0 3rem',
                zIndex: 6,
                animation: 'slideUp 0.8s ease 0.3s both',
            }}>
                {/* Headline */}
                <h1 style={{
                    color: 'white',
                    fontSize: '1.6rem',
                    fontWeight: 400,
                    lineHeight: 1.5,
                    marginBottom: '0.6rem',
                    fontFamily: 'Georgia, serif',
                    letterSpacing: '-0.01em',
                }}>
                    Fraud doesn't hide in reports.<br />
                    It hides in the <em style={{ color: '#C9A84C' }}>data between</em> reports.
                </h1>

                {/* Subtitle */}
                <div style={{
                    color: 'rgba(255,255,255,0.38)',
                    fontSize: '0.78rem',
                    fontFamily: 'Helvetica Neue, Arial, sans-serif',
                    textTransform: 'uppercase',
                    letterSpacing: '0.14em',
                    marginBottom: '1.25rem',
                }}>
                    Multimodal Industrial Fraud Detection — Algeria
                </div>

                {/* 4 AI model pills */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    {[
                        ['📡', 'LSTM + Isolation Forest'],
                        ['⚡', 'XGBoost Classifier'],
                        ['📝', 'BERT NLP'],
                        ['👁', 'YOLOv8 Vision'],
                    ].map(([icon, label]) => (
                        <div key={label} style={{
                            display: 'flex', alignItems: 'center', gap: '0.35rem',
                            padding: '0.3rem 0.7rem',
                            background: 'rgba(201,168,76,0.08)',
                            border: '1px solid rgba(201,168,76,0.18)',
                            borderRadius: 20,
                        }}>
                            <span style={{ fontSize: '0.7rem' }}>{icon}</span>
                            <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.65rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', letterSpacing: '0.04em' }}>{label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Footer */}
            <div style={{
                position: 'absolute', bottom: 20,
                color: 'rgba(255,255,255,0.18)', fontSize: '0.62rem',
                fontFamily: 'Helvetica Neue, Arial, sans-serif', textAlign: 'center',
                zIndex: 6, left: 0, right: 0,
                letterSpacing: '0.04em',
            }}>
                © 2026 FraudGuard AI · Université M'Hamed Bougara · Boumerdès
            </div>

            {/* Mobile fallback — hides canvas, shows static SVG badge */}
            <style>{`
                @media (max-width: 768px) {
                    .auth-left-panel-canvas { display: none !important; }
                }
            `}</style>
        </div>
    )
}