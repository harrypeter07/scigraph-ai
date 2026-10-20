// SciGraph 2.4 PRO - Main Application Bootstrap & View Orchestrator

async function loadViewTemplates() {
    const viewList = [
        'predictor.html',
        'teacher_verify.html',
        'verify_lab.html',
        'audit.html',
        'benchmarks.html',
        'compare.html',
        'topic_feasibility.html'
    ];
    
    const container = document.getElementById('viewsContainer');
    if (!container) return;
    
    try {
        const responses = await Promise.all(
            viewList.map(async (v) => {
                let res = await fetch(`/views/${v}?v=20261020-evidence1`);
                if (!res.ok) {
                    res = await fetch(`/static/views/${v}?v=20261020-evidence1`);
                }
                if (!res.ok) throw new Error(`HTTP ${res.status} fetching view ${v}`);
                return res.text();
            })
        );
        container.innerHTML = responses.join('\n');
        
        // Render any KaTeX math in the loaded views if auto-render is available
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(container, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false}
                ],
                throwOnError: false
            });
        }
    } catch (err) {
        console.error('Failed to load view templates:', err);
        container.innerHTML = `<div style="padding: 2rem; color: var(--destructive-red); font-family: var(--font-mono);">Failed to load application views: ${err.message}</div>`;
    }
}

window.addEventListener('DOMContentLoaded', async () => {
    // 1. Asynchronously load all modular view templates into the workspace container
    await loadViewTemplates();
    if (typeof loadForecastEvidence === 'function') loadForecastEvidence();
    document.getElementById('paperInput')?.addEventListener('keydown', event => { if (event.key === 'Enter') runPrediction(); });
    
    // 2. Standby on initial view (do not load data prior to user action)
    // User initiates analysis via 'Generate Forecast' or preset selection
    
    // 3. Preload background corpus and audit scoreboard
    if (typeof loadTeacherCorpus === 'function') {
        loadTeacherCorpus();
    }
    if (typeof loadTeacherScoreboardAndFooter === 'function') {
        loadTeacherScoreboardAndFooter();
    }
});
