// SciGraph 2.4 PRO - Cytoscape Subgraph & Citation Network Visualization

let cy = null;

function zoomGraph(factor) {
    if (cy) {
        cy.zoom(cy.zoom() * factor);
        cy.center();
    }
}

function fitGraph() {
    if (cy) {
        cy.fit(30);
    }
}

function focusGraphNode(nodeId) {
    if (!cy) return;
    const node = cy.$(`#${nodeId}`);
    if (node && node.length > 0) {
        cy.$(':selected').unselect();
        node.select();
        updateNodeInspector(node.data());
        cy.animate({
            center: { eles: node },
            zoom: Math.max(cy.zoom(), 1.15),
            duration: 300
        });
    }
}

function updateNodeInspector(d) {
    if (!d) return;

    const hEl = document.getElementById('inspHopBadge');
    const tEl = document.getElementById('inspType');
    const lEl = document.getElementById('inspLabel');
    const mEl = document.getElementById('inspMetricsGrid');
    const bBanner = document.getElementById('inspAttributionBanner');
    const aName = document.getElementById('inspAttributionName');
    const aBadge = document.getElementById('inspImpactBadge');
    const inText = document.getElementById('inspInsightText');

    const hopNum = d.hop != null ? d.hop : 0;
    if (hEl) {
        hEl.className = `hop-tag hop-${hopNum}`;
        hEl.textContent = d.hop_badge || `HOP ${hopNum}`;
    }

    safeSetText(tEl, d.role_desc || d.type || 'Entity');
    safeSetText(lEl, d.full_label || d.label || '—');

    // Live Model Attribution Banner
    if (bBanner && aName && aBadge) {
        if (d.attribution_name) {
            bBanner.style.display = 'flex';
            safeSetText(aName, d.attribution_name);
            const delta = d.attribution_delta || '0.0%';
            const dir = d.attribution_dir || 'neutral';
            aBadge.className = `insp-impact-badge ${dir}`;
            aBadge.textContent = `${delta} (${dir === 'positive' ? 'Boosts High Tier' : (dir === 'negative' ? 'Drags to Low Tier' : 'Neutral')})`;
        } else {
            bBanner.style.display = 'none';
        }
    }

    // 4-Metric Scientific Grid
    if (mEl) {
        if (d.metrics && Array.isArray(d.metrics) && d.metrics.length > 0) {
            mEl.innerHTML = d.metrics.map(m => `
                <div>
                    <div class="metric-cell-lbl">${m.label}</div>
                    <div class="metric-cell-val" title="${m.val}">${m.val}</div>
                </div>
            `).join('');
            mEl.style.display = 'grid';
        } else {
            mEl.style.display = 'none';
        }
    }

    // Algorithmic Decision Insight
    if (inText) {
        const narrative = d.insight || d.explanation || 'Structural node participating in multi-relational GNN message-passing aggregation.';
        inText.innerHTML = narrative;
    }
}

function renderSubgraph(data, containerId) {
    const container = document.getElementById(containerId || 'cy');
    if (!container || !data) return;
    const elements = [];

    // Extract authentic model attributions
    const attributions = Array.isArray(data.attributions) ? data.attributions : [];
    const findAttr = (kw) => attributions.find(a => (a.feature_name || '').toLowerCase().includes(kw)) || null;

    const citationAttr = findAttr('citation') || findAttr('momentum');
    const authorAttr = findAttr('author');
    const venueAttr = findAttr('venue');
    const lineageAttr = findAttr('lineage') || findAttr('disruption');
    const structAttr = findAttr('structural') || findAttr('graph') || findAttr('centrality');

    // Empirical cohort reference thresholds
    const bTest = data.backtest || {};
    const tProv = bTest.threshold_provenance || {};
    const p50 = typeof tProv.p50 === 'number' ? Math.round(tProv.p50) : 48;
    const p90 = typeof tProv.p90 === 'number' ? Math.round(tProv.p90) : 320;
    const citsCutoff = Number(data.citations_at_cutoff ?? data.cutoff_citations ?? 0);
    const pubYr = data.publication_year || 2024;
    const cutoffYr = data.cutoff_year || (pubYr + 2);
    const predLabel = data.predicted_impact_label || data.predicted_label || '—';
    const fieldName = data.primary_field || 'Computer Science';

    // 1. Target Paper (Hop 0)
    const cDelta = citationAttr ? `${citationAttr.direction === 'positive' ? '+' : '-'}${Math.abs(citationAttr.delta_percentage_points)}%` : '+0.0%';
    const cDir = citationAttr ? citationAttr.direction : 'neutral';

    let targetInsight = '';
    if (predLabel.includes('Low')) {
        targetInsight = `<strong>Decision Logic:</strong> With <strong>${citsCutoff.toLocaleString()} citations</strong> accumulated by cutoff (${cutoffYr}), this work performs below the field P90 threshold (<strong>${p90} cits</strong> in ${fieldName}). Despite early activity, the 2-year velocity confirms sub-outlier accumulation, focusing model probability into the Low (&lt;50th) tier.`;
    } else if (predLabel.includes('High')) {
        targetInsight = `<strong>Decision Logic:</strong> Accumulating <strong>${citsCutoff.toLocaleString()} citations</strong> by year ${cutoffYr} surpasses the empirical P90 threshold (<strong>${p90} cits</strong> in ${fieldName}). This rapid accumulation produces a <strong>${cDelta} probability lift</strong>, securing the High (≥90th) impact designation.`;
    } else {
        targetInsight = `<strong>Decision Logic:</strong> With <strong>${citsCutoff.toLocaleString()} citations</strong> by cutoff (${cutoffYr}), early trajectory tracks between the P50 (<strong>${p50}</strong>) and P90 (<strong>${p90}</strong>) percentiles, yielding a balanced Medium (50–90th) forecast.`;
    }

    elements.push({
        data: {
            id: 'target',
            label: data.title && data.title.length > 22 ? data.title.slice(0, 22) + '...' : (data.title || 'Seed Work'),
            full_label: data.title || 'Target Work',
            type: 'paper',
            hop: 0,
            hop_badge: 'HOP 0 (FOCAL SEED)',
            role_desc: 'Target Focal Paper (Predictive Root)',
            attribution_name: 'Citation Momentum & Early Growth',
            attribution_delta: cDelta,
            attribution_dir: cDir,
            metrics: [
                { label: 'Cutoff Citations', val: `${citsCutoff.toLocaleString()} (t ≤ ${cutoffYr})` },
                { label: 'Field P50 / P90', val: `${p50} / ${p90} cits` },
                { label: 'Predicted Tier', val: predLabel },
                { label: 'Split Invariance', val: data.in_train_split ? 'Train Set' : 'Held-Out Test' }
            ],
            insight: targetInsight
        }
    });

    // 2. Authors (Hop 1) and Institutions (Hop 2)
    const authorList = (data.author_details && data.author_details.length > 0)
        ? data.author_details
        : (data.authors || []).map((name, i) => ({
            name: name,
            position: i === 0 ? 'first' : (i === ((data.authors || []).length - 1) ? 'last' : 'middle'),
            institutions: []
        }));

    const aDelta = authorAttr ? `${authorAttr.direction === 'positive' ? '+' : '-'}${Math.abs(authorAttr.delta_percentage_points)}%` : '+0.0%';
    const aDir = authorAttr ? authorAttr.direction : 'neutral';

    authorList.slice(0, 3).forEach((auth, i) => {
        const aid = `auth_${i}`;
        const pos = auth.position || 'middle';
        const posLabel = pos === 'first' ? 'Lead Author (1st)' : (pos === 'last' ? 'Senior / PI Author' : 'Co-Author');

        const authInsight = `<strong>Message-Passing Flow:</strong> Author <strong>${auth.name}</strong> (${posLabel}) acts as a direct Hop 1 contributor. In HeteroSAGE, author historical citation velocity and co-authorship degree are aggregated with a <strong>${aDelta} feature attribution</strong>, providing foundational collaborative context.`;

        elements.push({
            data: {
                id: aid,
                label: auth.name.length > 18 ? auth.name.slice(0, 18) + '...' : auth.name,
                full_label: auth.name,
                type: 'author',
                hop: 1,
                hop_badge: 'HOP 1 (AUTHOR)',
                role_desc: `Collaborating Author (${posLabel})`,
                attribution_name: 'Author Track Record & Centrality',
                attribution_delta: aDelta,
                attribution_dir: aDir,
                metrics: [
                    { label: 'Author Name', val: auth.name },
                    { label: 'Authorship Role', val: posLabel },
                    { label: 'Graph Topology', val: 'Hop 1 Contributor' },
                    { label: 'Model Impact', val: aDelta }
                ],
                insight: authInsight
            }
        });
        elements.push({ data: { source: aid, target: 'target', label: 'authored' } });

        // Hop 2: Institutional Affiliation
        const insts = (auth.institutions && auth.institutions.length > 0) ? auth.institutions.slice(0, 1) : [];
        insts.forEach((instName, j) => {
            const iid = `inst_${i}_${j}`;
            const sDelta = structAttr ? `${structAttr.direction === 'positive' ? '+' : '-'}${Math.abs(structAttr.delta_percentage_points)}%` : '+0.0%';
            const sDir = structAttr ? structAttr.direction : 'neutral';

            elements.push({
                data: {
                    id: iid,
                    label: instName.length > 20 ? instName.slice(0, 20) + '...' : instName,
                    full_label: instName,
                    type: 'institution',
                    hop: 2,
                    hop_badge: 'HOP 2 (INSTITUTION)',
                    role_desc: 'Academic Institution (Hop 2)',
                    attribution_name: 'Institutional Ranking & Endowment',
                    attribution_delta: sDelta,
                    attribution_dir: sDir,
                    metrics: [
                        { label: 'Institution', val: instName },
                        { label: 'Affiliated Author', val: auth.name },
                        { label: 'Network Distance', val: 'Hop 2 (Structural Context)' },
                        { label: 'Prestige Flow', val: 'Inst → Author → Paper' }
                    ],
                    insight: `<strong>Structural Endowment:</strong> <strong>${instName}</strong> supplies secondary structural regularization at Hop 2. Institutional co-author networks channel research infrastructure weight into the primary author embedding during GNN convolution.`
                }
            });
            elements.push({ data: { source: aid, target: iid, label: 'affiliated_with' } });
        });
    });

    // 3. Publication Venue (Hop 1)
    if (data.venue) {
        const v = data.venue;
        const vName = v.venue_name || 'Academic Venue';
        const vTier = v.quartile || v.core_rank || 'Peer-Reviewed';
        const vDelta = venueAttr ? `${venueAttr.direction === 'positive' ? '+' : '-'}${Math.abs(venueAttr.delta_percentage_points)}%` : '+0.0%';
        const vDir = venueAttr ? venueAttr.direction : 'neutral';

        const isQ1 = (v.quartile === 'Q1' || v.core_rank === 'A*' || v.core_rank === 'A');
        const vInsight = isQ1
            ? `<strong>Prestige Multiplier:</strong> Published in <strong>${vName}</strong> (${v.quartile || 'Elite'}, SJR ${v.sjr_score ? Number(v.sjr_score).toFixed(2) : 'Cataloged'}). Verified editorial selectivity provides a <strong>${vDelta} positive boost</strong> to high-impact likelihood.`
            : `<strong>Standard Venue Context:</strong> <strong>${vName}</strong> (${v.quartile || 'Indexed'}, SJR ${v.sjr_score ? Number(v.sjr_score).toFixed(2) : '0.0'}) provides standard dissemination without outlier inflation, focusing model reliance on paper-level citation velocity.`;

        elements.push({
            data: {
                id: 'venue_node',
                label: vName.length > 20 ? vName.slice(0, 20) + '...' : vName,
                full_label: `${vName} (${vTier})`,
                type: 'venue',
                hop: 1,
                hop_badge: 'HOP 1 (VENUE)',
                role_desc: `Publication Venue (${vTier})`,
                attribution_name: 'Venue Prestige (SCImago & CORE)',
                attribution_delta: vDelta,
                attribution_dir: vDir,
                metrics: [
                    { label: 'SCImago Quartile', val: v.quartile || 'Unranked' },
                    { label: 'SJR Score', val: v.sjr_score ? Number(v.sjr_score).toFixed(2) : 'Cataloged' },
                    { label: 'CORE Conf Rank', val: v.core_rank || 'N/A' },
                    { label: 'Model Impact', val: vDelta }
                ],
                insight: vInsight
            }
        });
        elements.push({ data: { source: 'target', target: 'venue_node', label: 'published_in' } });
    }

    // 4. Topic / Discipline (Hop 1)
    if (data.primary_topic) {
        elements.push({
            data: {
                id: 'topic_node',
                label: data.primary_topic.length > 20 ? data.primary_topic.slice(0, 20) + '...' : data.primary_topic,
                full_label: data.primary_topic,
                type: 'topic',
                hop: 1,
                hop_badge: 'HOP 1 (TOPIC)',
                role_desc: 'Discipline & Topic Anchor',
                attribution_name: 'Topic / Field Velocity Normalization',
                attribution_delta: 'Calibrated',
                attribution_dir: 'neutral',
                metrics: [
                    { label: 'Topic Concept', val: data.primary_topic },
                    { label: 'Discipline', val: fieldName },
                    { label: 'Field P90 Threshold', val: `${p90} citations` },
                    { label: 'Dynamics Mode', val: 'Log-Growth Normalized' }
                ],
                insight: `<strong>Field Normalization:</strong> In <strong>${fieldName}</strong>, empirical citation cutpoints require <strong>${p90} citations</strong> to enter the 90th percentile, controlling for disciplinary citation velocity differences.`
            }
        });
        elements.push({ data: { source: 'target', target: 'topic_node', label: 'has_topic' } });
    }

    // 5. Precursor References (Hop 1)
    const outflow = (data.lineage && Array.isArray(data.lineage.outflow_sample)) ? data.lineage.outflow_sample : [];
    const cdVal = Number((data.lineage && data.lineage.cd_index_5) || 0);
    const lDelta = lineageAttr ? `${lineageAttr.direction === 'positive' ? '+' : '-'}${Math.abs(lineageAttr.delta_percentage_points)}%` : '+0.0%';
    const lDir = lineageAttr ? lineageAttr.direction : 'neutral';

    outflow.slice(0, 3).forEach((r, idx) => {
        const rid = `ref_${idx}`;
        const refTitle = r.title || `Foundational Reference ${idx + 1}`;
        const isConsolidating = cdVal <= 0;

        const refInsight = `<strong>Lineage Impact:</strong> Cites precursor work <em>"${refTitle}"</em> (${(r.citation_count || 0).toLocaleString()} cits). The Wu et al. CD index of <strong>${cdVal.toFixed(4)}</strong> reflects a <strong>${isConsolidating ? 'consolidating paradigm' : 'disruptive branch'}</strong>, shifting the lineage attribution by <strong>${lDelta}</strong>.`;

        elements.push({
            data: {
                id: rid,
                label: refTitle.length > 18 ? refTitle.slice(0, 18) + '...' : refTitle,
                full_label: refTitle,
                type: 'precursor',
                hop: 1,
                hop_badge: 'HOP 1 (PRECURSOR REF)',
                role_desc: 'Foundational Precursor Reference',
                attribution_name: 'Lineage & Disruption (CD Index)',
                attribution_delta: lDelta,
                attribution_dir: lDir,
                metrics: [
                    { label: 'Precursor Work', val: refTitle.slice(0, 24) + '...' },
                    { label: 'Pub Year', val: String(r.publication_year || 'Prior') },
                    { label: 'Wu et al. CD Index', val: cdVal.toFixed(4) },
                    { label: 'Topology Impact', val: isConsolidating ? 'Consolidating' : 'Disruptive' }
                ],
                insight: refInsight
            }
        });
        elements.push({ data: { source: 'target', target: rid, label: 'cites' } });
    });

    // Populate Connected Neighbors Influence Breakdown Table
    const tableBody = document.getElementById('connectedImpactTableBody');
    if (tableBody) {
        const influenceRows = [
            {
                id: 'target',
                entity: data.title ? (data.title.slice(0, 30) + '...') : 'Seed Paper',
                type: 'Seed (Hop 0)',
                signal: `${citsCutoff} cutoff cits (t ≤ ${cutoffYr})`,
                delta: cDelta,
                dir: cDir
            }
        ];
        if (data.venue) {
            influenceRows.push({
                id: 'venue_node',
                entity: data.venue.venue_name ? (data.venue.venue_name.slice(0, 28) + '...') : 'Venue',
                type: 'Venue (Hop 1)',
                signal: `${data.venue.quartile || 'Indexed'} (SJR: ${data.venue.sjr_score ? Number(data.venue.sjr_score).toFixed(2) : '—'})`,
                delta: venueAttr ? `${venueAttr.direction === 'positive' ? '+' : '-'}${Math.abs(venueAttr.delta_percentage_points)}%` : '+0.0%',
                dir: venueAttr ? venueAttr.direction : 'neutral'
            });
        }
        if (authorList.length > 0) {
            influenceRows.push({
                id: 'auth_0',
                entity: authorList[0].name || 'Lead Author',
                type: 'Author (Hop 1)',
                signal: `${authorList[0].position || 'Lead'} Contributor`,
                delta: aDelta,
                dir: aDir
            });
        }
        if (data.primary_topic) {
            influenceRows.push({
                id: 'topic_node',
                entity: data.primary_topic.slice(0, 28),
                type: 'Topic (Hop 1)',
                signal: `${fieldName} (P90: ${p90})`,
                delta: 'Calibrated',
                dir: 'neutral'
            });
        }
        if (outflow.length > 0) {
            influenceRows.push({
                id: 'ref_0',
                entity: (outflow[0].title || 'Precursor Work').slice(0, 28),
                type: 'Reference (Hop 1)',
                signal: `CD Index: ${cdVal.toFixed(4)}`,
                delta: lDelta,
                dir: lDir
            });
        }

        tableBody.innerHTML = influenceRows.map(r => `
            <tr onclick="focusGraphNode('${r.id}')" title="Click to inspect this entity">
                <td style="font-weight: 600; color: var(--accent-blue);">${r.entity}</td>
                <td><span style="font-family: var(--font-mono); font-size: 0.68rem;">${r.type}</span></td>
                <td>${r.signal}</td>
                <td><span class="insp-impact-badge ${r.dir}">${r.delta}</span></td>
            </tr>
        `).join('');
    }

    if (container._cy) {
        try { container._cy.destroy(); } catch (e) {}
    }
    if (!containerId && cy) {
        try { cy.destroy(); } catch (e) {}
    }

    const graph = cytoscape({
        container: container,
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'color': '#F1F5F9',
                    'font-size': '10px',
                    'font-weight': '600',
                    'font-family': 'Inter, sans-serif',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'text-outline-color': '#080B11',
                    'text-outline-width': 2,
                    'background-color': '#10B981',
                    'width': 28,
                    'height': 28,
                    'border-width': 2,
                    'border-color': '#34D399'
                }
            },
            { selector: 'node[type = "paper"]', style: { 'background-color': '#3B82F6', 'width': 42, 'height': 42, 'border-width': 3, 'border-color': '#93C5FD' } },
            { selector: 'node[type = "author"]', style: { 'background-color': '#6366F1', 'width': 26, 'height': 26, 'border-color': '#A5B4FC' } },
            { selector: 'node[type = "institution"]', style: { 'background-color': '#0D9488', 'width': 22, 'height': 22, 'border-color': '#5EEAD4' } },
            { selector: 'node[type = "venue"]', style: { 'background-color': '#F59E0B', 'width': 30, 'height': 30, 'border-color': '#FDE68A' } },
            { selector: 'node[type = "topic"]', style: { 'background-color': '#10B981', 'width': 28, 'height': 28, 'border-color': '#6EE7B7' } },
            { selector: 'node[type = "precursor"]', style: { 'background-color': '#8B5CF6', 'width': 24, 'height': 24, 'border-color': '#C4B5FD' } },
            {
                selector: 'node:selected',
                style: {
                    'border-width': 4,
                    'border-color': '#38BDF8',
                    'shadow-blur': 16,
                    'shadow-color': 'rgba(56, 189, 248, 0.7)',
                    'shadow-opacity': 0.9
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 1.6,
                    'line-color': '#2A364F',
                    'target-arrow-color': '#475569',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 0.85
                }
            },
            { selector: 'edge[label = "authored"]', style: { 'line-color': '#3B82F6', 'target-arrow-color': '#60A5FA', 'opacity': 0.75 } },
            { selector: 'edge[label = "affiliated_with"]', style: { 'line-color': '#14B8A6', 'line-style': 'dotted', 'opacity': 0.65 } },
            { selector: 'edge[label = "published_in"]', style: { 'line-color': '#F59E0B', 'target-arrow-color': '#FBBF24', 'opacity': 0.75 } },
            { selector: 'edge[label = "has_topic"]', style: { 'line-color': '#10B981', 'line-style': 'dashed', 'opacity': 0.65 } },
            { selector: 'edge[label = "cites"]', style: { 'line-color': '#8B5CF6', 'target-arrow-color': '#A78BFA', 'opacity': 0.75 } }
        ],
        layout: {
            name: 'cose',
            animate: false,
            padding: 30,
            nodeRepulsion: 18000,
            idealEdgeLength: 85,
            edgeElasticity: 100,
            gravity: 80,
            numIter: 1000
        }
    });

    graph.on('tap', 'node', function(evt) {
        updateNodeInspector(evt.target.data());
    });
    container._cy = graph;
    if (!containerId) cy = graph;

    setTimeout(() => {
        graph.resize();
        graph.fit(30);
        const targetNode = graph.$('#target');
        if (targetNode.length > 0) {
            targetNode.select();
            updateNodeInspector(targetNode.data());
        }
    }, 120);
}
