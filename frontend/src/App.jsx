import { Component, useEffect, useState } from 'react'
import { NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Activity, Archive, BarChart3, Bot, ChevronDown, Command, FileText, LayoutDashboard, Menu, Moon, Search, Settings, Sun, Clock3, X } from 'lucide-react'
import { Dashboard, Chat, Memories, Documents, AdvancedSearch, Timeline, Analytics, SystemStatus, Preferences } from './pages/Pages.jsx'

const nav = [
  { label: 'Overview', items: [['Dashboard', '/', LayoutDashboard], ['AI Chat', '/chat', Bot]] },
  { label: 'Your knowledge', items: [['Memory Explorer', '/memories', Archive], ['Documents', '/documents', FileText], ['Advanced Search', '/search', Search], ['Timeline', '/timeline', Clock3]] },
  { label: 'Workspace', items: [['Analytics', '/analytics', BarChart3], ['System Status', '/system', Activity], ['Settings', '/settings', Settings]] },
]
const titles = { '/': ['Your knowledge, in focus', 'Dashboard'], '/chat': ['Ask across your knowledge base', 'AI Chat'], '/memories': ['Browse the indexed knowledge base', 'Memory Explorer'], '/documents': ['Manage your source library', 'Documents'], '/search': ['Tune into the right memory', 'Advanced Search'], '/timeline': ['Explore when things happened', 'Timeline'], '/analytics': ['Measure retrieval quality', 'Analytics / Evaluation'], '/system': ['Runtime and index diagnostics', 'System Status'], '/settings': ['Personalize your workspace', 'Settings'] }
const SHOW_PAGE_DIAGNOSTICS = import.meta.env.DEV

class PageErrorBoundary extends Component {
  state = { error: null, componentStack: '' }
  static getDerivedStateFromError(error) { return { error } }
  componentDidCatch(error, info) {
    console.error(`MemoraAI page render failed at ${this.props.resetKey}`, error, info.componentStack)
    this.setState({ componentStack: info.componentStack || '' })
  }
  componentDidUpdate(previous) {
    if (previous.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null, componentStack: '' })
  }
  render() {
    if (this.state.error) return <section className="panel page-error" role="alert"><div className="eyebrow">PAGE RENDER ERROR · {this.props.resetKey}</div><h2>{this.state.error.message || String(this.state.error)}</h2>{SHOW_PAGE_DIAGNOSTICS && <div className="page-error-diagnostics"><h3>Component stack</h3><pre>{this.state.componentStack || '(component stack unavailable)'}</pre><h3>JavaScript stack</h3><pre>{this.state.error.stack || '(JavaScript stack unavailable)'}</pre></div>}<button className="button button-secondary" onClick={() => this.setState({ error: null, componentStack: '' })}>Retry render</button></section>
    return this.props.children
  }
}

export default function App() {
  const [theme, setTheme] = useState(localStorage.getItem('memora-theme') || 'dark')
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('memora-theme', theme) }, [theme])
  useEffect(() => { setMobileOpen(false) }, [location.pathname])
  const [eyebrow, title] = titles[location.pathname] || titles['/']
  const pages = <Routes>
    <Route path="/" element={<Dashboard />} /><Route path="/chat" element={<Chat />} /><Route path="/memories" element={<Memories />} />
    <Route path="/documents" element={<Documents />} /><Route path="/search" element={<AdvancedSearch />} /><Route path="/timeline" element={<Timeline />} />
    <Route path="/analytics" element={<Analytics />} /><Route path="/system" element={<SystemStatus />} /><Route path="/settings" element={<Preferences theme={theme} setTheme={setTheme} />} />
  </Routes>
  return <div className="app-shell">
    {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Command size={19}/></div><div><strong>memora<span>ai</span></strong><small>PERSONAL INTELLIGENCE</small></div><button className="icon-btn sidebar-close" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={18}/></button></div>
      <div className="workspace-switch"><div className="avatar">M</div><div><b>My workspace</b><small>Local knowledge base</small></div><ChevronDown size={15}/></div>
      <nav>{nav.map(section => <div className="nav-section" key={section.label}><div className="nav-label">{section.label}</div>{section.items.map(([label, path, Icon]) => <NavLink end={path === '/'} to={path} key={path} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}><Icon size={17}/><span>{label}</span>{path === '/chat' && <i className="live-dot"/>}</NavLink>)}</div>)}</nav>
      <div className="sidebar-bottom"><div className="local-badge"><span className="status-dot"/> All data stays on this device <i>LOCAL</i></div><div className="user-row"><div className="avatar avatar-user">M</div><div><b>My account</b><small>Personal workspace</small></div><button className="icon-btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>}</button></div></div>
    </aside>
    <main className="main-area"><header className="topbar"><button className="icon-btn mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open menu"><Menu size={19}/></button><div className="breadcrumbs"><span>Workspace</span><b>/</b><strong>{title}</strong></div><div className="top-actions"><div className="api-indicator"><span className="status-dot"/> Local API</div><button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle color theme">{theme === 'dark' ? <Sun size={16}/> : <Moon size={16}/>}</button><div className="avatar avatar-top">M</div></div></header>
      <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1></div><div className="heading-date"><span className="status-dot"/> Private workspace</div></div>
      <AnimatePresence mode="wait"><motion.div key={location.pathname} className="page-content" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: .2 }}><PageErrorBoundary resetKey={location.pathname}>{pages}</PageErrorBoundary></motion.div></AnimatePresence>
      <footer className="footer"><span>MEMORAAI <i>•</i> LOCAL FIRST AI</span><span>Your knowledge belongs to you.</span></footer>
    </main>
  </div>
}
