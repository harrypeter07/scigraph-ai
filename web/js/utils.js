// SciGraph 2.4 PRO - Utility Primitives
// Formatting, DOM helpers, graph controls

        function fmtNum(n) {
            if (n === null || n === undefined || n === '' || Number.isNaN(Number(n))) return '—';
            return Number(n).toLocaleString();
        }

        function safeSetText(id, val) {
            const el = (typeof id === 'string') ? document.getElementById(id) : id;
            if (el) el.textContent = (val !== null && val !== undefined) ? String(val) : '—';
        }

        function safeSetHtml(id, val) {
            const el = (typeof id === 'string') ? document.getElementById(id) : id;
            if (el) el.innerHTML = (val !== null && val !== undefined) ? String(val) : '';
        }

        function cutoffCount(d) {
            if (!d) return null;
            if (d.citations_at_cutoff != null) return d.citations_at_cutoff;
            if (d.cutoff_citations != null) return d.cutoff_citations;
            return null;
        }

        function setTierBadge(el, label, cls) {
            if (!el) return;
            el.className = 'outcome-badge';
            if (cls === 2 || (label && label.toLowerCase().includes('high'))) el.classList.add('high');
            else if (cls === 1 || (label && label.toLowerCase().includes('medium'))) el.classList.add('medium');
            else el.classList.add('low');
            el.textContent = label || '—';
        }

        function zoomGraph(factor) {
            if (!cy) return;
            cy.zoom(cy.zoom() * factor);
            cy.center();
        }

        function fitGraph() {
            if (!cy) return;
            cy.resize();
            cy.fit(30);
        }

