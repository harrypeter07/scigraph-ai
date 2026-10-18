// SciGraph 2.4 PRO - Cytoscape Subgraph & Citation Network Visualization

        let cy = null;

        function updateNodeInspector(d) {
            const hEl = document.getElementById('inspHopBadge');
            const tEl = document.getElementById('inspType');
            const lEl = document.getElementById('inspLabel');
            const mEl = document.getElementById('inspMetricsGrid');
            const dEl = document.getElementById('inspDetail');

            const hopNum = d.hop != null ? d.hop : 0;
            if (hEl) {
                hEl.className = `hop-tag hop-${hopNum}`;
                hEl.textContent = d.hop_badge || `HOP ${hopNum}`;
            }

            safeSetText(tEl, d.role_desc || d.type || 'Entity');
            safeSetText(lEl, d.full_label || d.label);
            safeSetText(dEl, d.explanation || 'Structural node in 2-hop message-passing network.');

            if (mEl) {
                if (d.metrics && Array.isArray(d.metrics)) {
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
        }

        function renderSubgraph(data, containerId) {
            const container = document.getElementById(containerId || 'cy');
            if (!container || !data) return;
            const elements = [];

            // 1. Target Paper (Hop 0)
            const targetMetrics = [
                { label: 'Cutoff Citations', val: fmtNum(cutoffCount(data)) + ' (T_cutoff)' },
                { label: 'Pub Year', val: String(data.publication_year || 'N/A') },
                { label: 'Forecast Tier', val: data.predicted_impact_label || data.predicted_label || '—' },
                { label: 'Split', val: data.in_train_split ? 'Train (2012–2016)' : 'Held-Out Test' }
            ];

            elements.push({
                data: {
                    id: 'target',
                    label: data.title.length > 24 ? data.title.slice(0, 24) + '...' : data.title,
                    full_label: data.title,
                    type: 'paper',
                    hop: 0,
                    hop_badge: 'HOP 0 (FOCAL SEED)',
                    role_desc: 'Target Focal Paper (Predictive Root)',
                    metrics: targetMetrics,
                    explanation: 'This is the focal paper being evaluated. In our 2-Hop Heterogeneous GNN (HeteroSAGE), this node aggregates citations, author reputations, institutional rankings, and venue prestige to forecast citations in years 3-5 after publication.'
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

            authorList.slice(0, 4).forEach((auth, i) => {
                const aid = `auth_${i}`;
                const pos = auth.position || 'middle';
                const posLabel = pos === 'first' ? 'Lead Author (1st)' : (pos === 'last' ? 'Senior / PI Author' : 'Co-Author');

                const authMetrics = [
                    { label: 'Author Name', val: auth.name },
                    { label: 'Position', val: posLabel },
                    { label: 'Network Distance', val: 'Hop 1 (Direct Contributor)' },
                    { label: 'Prestige Flow', val: 'Direct to Seed Paper' }
                ];

                elements.push({
                    data: {
                        id: aid,
                        label: auth.name.length > 18 ? auth.name.slice(0, 18) + '...' : auth.name,
                        full_label: auth.name,
                        type: 'author',
                        hop: 1,
                        hop_badge: 'HOP 1 (AUTHOR)',
                        role_desc: `Collaborating Author (${posLabel})`,
                        metrics: authMetrics,
                        explanation: 'Who wrote the paper? In scientometrics, an author’s prior track record and collaboration network provide strong early signals before citations accumulate.'
                    }
                });
                elements.push({ data: { source: aid, target: 'target', label: 'authored' } });

                // Hop 2: Institutional Affiliation
                const insts = (auth.institutions && auth.institutions.length > 0) ? auth.institutions.slice(0, 1) : [];
                insts.forEach((instName, j) => {
                    const iid = `inst_${i}_${j}`;
                    const instMetrics = [
                        { label: 'Institution', val: instName },
                        { label: 'Affiliated Author', val: auth.name },
                        { label: 'Network Distance', val: 'Hop 2 (Structural Context)' },
                        { label: 'Prestige Flow', val: 'Inst → Author → Paper' }
                    ];

                    elements.push({
                        data: {
                            id: iid,
                            label: instName.length > 20 ? instName.slice(0, 20) + '...' : instName,
                            full_label: instName,
                            type: 'institution',
                            hop: 2,
                            hop_badge: 'HOP 2 (INSTITUTION)',
                            role_desc: 'Institutional Affiliation',
                            metrics: instMetrics,
                            explanation: 'Where was the research conducted? Elite universities and labs offer advanced equipment, grant funding, and international recognition.'
                        }
                    });
                    elements.push({ data: { source: aid, target: iid, label: 'affiliated_with' } });
                });
            });

            // 3. Venue (Hop 1)
            if (data.venue) {
                const vName = data.venue.venue_name || 'Academic Venue';
                const vTier = data.venue.quartile || data.venue.core_rank || 'Peer-Reviewed';
                const vMetrics = [
                    { label: 'Venue Name', val: vName },
                    { label: 'SCImago Quartile', val: data.venue.quartile || 'Unranked' },
                    { label: 'CORE Rank', val: data.venue.core_rank || 'N/A' },
                    { label: 'SJR Score', val: data.venue.sjr_score ? Number(data.venue.sjr_score).toFixed(2) : 'Cataloged' }
                ];

                elements.push({
                    data: {
                        id: 'venue_node',
                        label: vName.length > 20 ? vName.slice(0, 20) + '...' : vName,
                        full_label: `${vName} (${vTier})`,
                        type: 'venue',
                        hop: 1,
                        hop_badge: 'HOP 1 (VENUE)',
                        role_desc: `Publication Venue (${vTier})`,
                        metrics: vMetrics,
                        explanation: 'Where was it published? Prestigious peer-reviewed venues (Q1 / CORE A*) act as rigorous quality certifications that attract early citations.'
                    }
                });
                elements.push({ data: { source: 'target', target: 'venue_node', label: 'published_in' } });
            }

            // 4. Topic (Hop 1)
            if (data.primary_topic) {
                const topicMetrics = [
                    { label: 'Field Concept', val: data.primary_topic },
                    { label: 'Parent Discipline', val: data.primary_field || 'Computer Science' },
                    { label: 'Citation Velocity', val: 'Field-Normalised' },
                    { label: 'Graph Role', val: 'Semantic Anchor' }
                ];

                elements.push({
                    data: {
                        id: 'topic_node',
                        label: data.primary_topic.length > 20 ? data.primary_topic.slice(0, 20) + '...' : data.primary_topic,
                        full_label: data.primary_topic,
                        type: 'topic',
                        hop: 1,
                        hop_badge: 'HOP 1 (TOPIC)',
                        role_desc: 'Primary Academic Topic',
                        metrics: topicMetrics,
                        explanation: 'What is the subject matter? Citation velocity differs widely across fields (e.g. AI vs Pure Math). The topic node normalises citation growth curves.'
                    }
                });
                elements.push({ data: { source: 'target', target: 'topic_node', label: 'has_topic' } });
            }

            // 5. Precursor References (Hop 1)
            (data.lineage && data.lineage.outflow_sample ? data.lineage.outflow_sample : []).slice(0, 3).forEach((r, idx) => {
                const rid = `ref_${idx}`;
                const refMetrics = [
                    { label: 'Precursor Title', val: r.title },
                    { label: 'Pub Year', val: String(r.publication_year || 'Prior') },
                    { label: 'Total Citations', val: (r.citation_count || 0).toLocaleString() },
                    { label: 'Disruption Impact', val: 'Outflow Foundational Basis' }
                ];

                elements.push({
                    data: {
                        id: rid,
                        label: r.title.length > 18 ? r.title.slice(0, 18) + '...' : r.title,
                        full_label: r.title,
                        type: 'precursor',
                        hop: 1,
                        hop_badge: 'HOP 1 (PRECURSOR REF)',
                        role_desc: 'Foundational Precursor Reference',
                        metrics: refMetrics,
                        explanation: 'What prior science did this paper cite? Citations to precursor works versus new citing works determine the Wu et al. CD Disruption Index.'
                    }
                });
                elements.push({ data: { source: 'target', target: rid, label: 'cites' } });
            });

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
                            'color': '#202522',
                            'font-size': '10px',
                            'font-weight': '500',
                            'font-family': 'Inter, sans-serif',
                            'text-valign': 'bottom',
                            'text-margin-y': 4,
                            'background-color': '#246B50',
                            'width': 28,
                            'height': 28,
                            'border-width': 1.5,
                            'border-color': '#FFFFFF'
                        }
                    },
                    { selector: 'node[type = "paper"]', style: { 'background-color': '#0284c7', 'width': 44, 'height': 44, 'border-width': 2.5, 'border-color': '#0369a1' } },
                    { selector: 'node[type = "author"]', style: { 'background-color': '#3B82F6', 'width': 26, 'height': 26 } },
                    { selector: 'node[type = "institution"]', style: { 'background-color': '#0D9488', 'width': 20, 'height': 20 } },
                    { selector: 'node[type = "venue"]', style: { 'background-color': '#D97706', 'width': 30, 'height': 30 } },
                    { selector: 'node[type = "topic"]', style: { 'background-color': '#16A34A', 'width': 28, 'height': 28 } },
                    { selector: 'node[type = "precursor"]', style: { 'background-color': '#7C3AED', 'width': 24, 'height': 24 } },
                    {
                        selector: 'edge',
                        style: {
                            'width': 1.4,
                            'line-color': '#D1D5DB',
                            'target-arrow-color': '#9CA3AF',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'arrow-scale': 0.8
                        }
                    },
                    { selector: 'edge[label = "authored"]', style: { 'line-color': '#93C5FD', 'target-arrow-color': '#60A5FA' } },
                    { selector: 'edge[label = "affiliated_with"]', style: { 'line-color': '#99F6E4', 'line-style': 'dotted' } },
                    { selector: 'edge[label = "published_in"]', style: { 'line-color': '#FDE68A' } },
                    { selector: 'edge[label = "has_topic"]', style: { 'line-color': '#BBF7D0', 'line-style': 'dashed' } },
                    { selector: 'edge[label = "cites"]', style: { 'line-color': '#DDD6FE' } }
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
            }, 100);
        }
