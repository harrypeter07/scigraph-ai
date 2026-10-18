// SciGraph 2.4 PRO - Topic Feasibility & Success Explorer Logic

let lastLoadedTopicData = null;

        async function loadTopicPresets() {
            try {
                const resp = await fetch('/api/v1/topic/presets');
                if (!resp.ok) return;
                const data = await resp.json();
                const container = document.getElementById('topicPresetsList');
                if (!container || !data.presets) return;

                container.innerHTML = '';
                data.presets.forEach(p => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'topic-pill-btn';
                    btn.textContent = p.title;
                    btn.onclick = () => selectTopicPreset(p.title);
                    container.appendChild(btn);
                });
            } catch (e) {
                // Keep default HTML presets if offline
            }
        }

        function selectTopicPreset(topicName) {
            const input = document.getElementById('topicSearchInput');
            if (input) {
                input.value = topicName;
                evaluateTopicFromInput();
            }
        }

        async function evaluateTopicFromInput() {
            const input = document.getElementById('topicSearchInput');
            const errCard = document.getElementById('topicErrorCard');
            const errMsg = document.getElementById('topicErrorMessage');
            const loadingCard = document.getElementById('topicLoadingCard');
            const resultsArea = document.getElementById('topicResultsArea');
            const btn = document.getElementById('btnEvaluateTopic');

            if (!input) return;
            const query = input.value.trim();
            if (!query) {
                if (errCard && errMsg) {
                    errMsg.textContent = 'Please enter a keyword, discipline, or research topic to evaluate.';
                    errCard.style.display = 'block';
                }
                return;
            }

            if (errCard) errCard.style.display = 'none';
            if (resultsArea) resultsArea.style.display = 'none';
            if (loadingCard) loadingCard.style.display = 'block';
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Evaluating...';
            }

            try {
                const url = `/api/v1/topic/explore?query=${encodeURIComponent(query)}&mode=${currentTopicMode}`;
                const resp = await fetch(url);
                if (!resp.ok) {
                    const errJson = await resp.json().catch(() => ({}));
                    throw new Error(errJson.detail || `Server returned HTTP ${resp.status}`);
                }
                const data = await resp.json();
                lastLoadedTopicData = data;
                renderTopicFeasibility(data);
                if (resultsArea) resultsArea.style.display = 'block';
            } catch (err) {
                if (errCard && errMsg) {
                    errMsg.textContent = err.message || 'Failed to crawl and evaluate topic.';
                    errCard.style.display = 'block';
                }
            } finally {
                if (loadingCard) loadingCard.style.display = 'none';
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Evaluate Feasibility';
                }
            }
        }

        function renderTopicFeasibility(data) {
            const fm = data.feasibility_metrics || {};
            const ti = data.topic_info || {};
            const ga = data.gemini_analysis || {};
            const pillars = fm.pillars || {};

            // 1. Success Rate Dial & Badges
            const rateNum = document.getElementById('topicSuccessRateNum');
            const rateTier = document.getElementById('topicSuccessRateTier');
            const feasBadge = document.getElementById('topicFeasibilityBadge');

            if (rateNum) rateNum.textContent = (fm.success_rate_pct !== undefined ? fm.success_rate_pct : '--') + '%';
            if (rateTier) rateTier.textContent = fm.recommendation_tier || 'Analyzed';
            if (feasBadge) {
                feasBadge.textContent = ti.feasibility_badge || 'Active Topic';
                feasBadge.className = 'outcome-badge ' + (fm.success_rate_pct >= 80 ? 'high' : (fm.success_rate_pct >= 65 ? 'medium' : 'low'));
            }

            // 2. Topic Taxonomy Header
            const dName = document.getElementById('topicDisplayName');
            const fPill = document.getElementById('topicFieldPill');
            const sfPill = document.getElementById('topicSubfieldPill');
            const desc = document.getElementById('topicDescriptionText');
            const totalPapers = document.getElementById('topicTotalPapersVal');
            const cagrVal = document.getElementById('topicCagrVal');
            const satVal = document.getElementById('topicSaturationVal');

            if (dName) dName.textContent = ti.display_name || data.query;
            if (fPill) fPill.textContent = ti.field || 'Multidisciplinary';
            if (sfPill) sfPill.textContent = ti.subfield || ti.domain || '';
            if (desc) desc.textContent = ti.description || '';
            if (totalPapers) totalPapers.textContent = (ti.total_works_indexed || 0).toLocaleString();
            if (cagrVal) cagrVal.textContent = (ti.cagr_3yr_pct !== undefined ? (ti.cagr_3yr_pct > 0 ? '+' : '') + ti.cagr_3yr_pct + '%' : '--');
            if (satVal) satVal.textContent = ti.saturation_state || '--';

            // 3. Four Pillars
            const gScore = document.getElementById('pillarGrowthScore');
            const gSumm = document.getElementById('pillarGrowthSummary');
            if (gScore && pillars.growth_velocity) gScore.textContent = Math.round(pillars.growth_velocity.score) + '/100';
            if (gSumm && pillars.growth_velocity) gSumm.textContent = pillars.growth_velocity.summary;

            const cScore = document.getElementById('pillarCitationScore');
            const cSumm = document.getElementById('pillarCitationSummary');
            if (cScore && pillars.citation_intensity) cScore.textContent = Math.round(pillars.citation_intensity.score) + '/100';
            if (cSumm && pillars.citation_intensity) cSumm.textContent = pillars.citation_intensity.summary;

            const vScore = document.getElementById('pillarVenueScore');
            const vSumm = document.getElementById('pillarVenueSummary');
            if (vScore && pillars.venue_accessibility) vScore.textContent = Math.round(pillars.venue_accessibility.score) + '/100';
            if (vSumm && pillars.venue_accessibility) vSumm.textContent = pillars.venue_accessibility.summary;

            const oScore = document.getElementById('pillarOAScore');
            const oSumm = document.getElementById('pillarOASummary');
            if (oScore && pillars.oa_dissemination) oScore.textContent = Math.round(pillars.oa_dissemination.score) + '/100';
            if (oSumm && pillars.oa_dissemination) oSumm.textContent = pillars.oa_dissemination.summary;

            // 4. Yearly Trajectory Bars
            const chartContainer = document.getElementById('topicTrajectoryBars');
            if (chartContainer) {
                chartContainer.innerHTML = '';
                const traj = data.yearly_trajectory || [];
                const maxCount = Math.max(...traj.map(t => t.count), 1);

                traj.forEach(item => {
                    const col = document.createElement('div');
                    col.className = 'trajectory-bar-col';

                    const countSpan = document.createElement('span');
                    countSpan.className = 'trajectory-bar-count';
                    countSpan.textContent = item.count >= 1000 ? Math.round(item.count / 1000) + 'k' : String(item.count);

                    const bar = document.createElement('div');
                    bar.className = 'trajectory-bar';
                    const heightPct = Math.max(10, Math.round((item.count / maxCount) * 100));
                    bar.style.height = heightPct + '%';
                    bar.title = `Year ${item.year}: ${item.count.toLocaleString()} works`;

                    const labelSpan = document.createElement('span');
                    labelSpan.className = 'trajectory-bar-label';
                    labelSpan.textContent = String(item.year);

                    col.appendChild(countSpan);
                    col.appendChild(bar);
                    col.appendChild(labelSpan);
                    chartContainer.appendChild(col);
                });
            }

            // 5. Gemini Intelligence Synthesis
            const execText = document.getElementById('topicExecutiveSummaryText');
            const gapsList = document.getElementById('topicNovelGapsList');
            const methodsList = document.getElementById('topicMethodsList');
            const pitfallsList = document.getElementById('topicPitfallsList');
            const modePill = document.getElementById('topicAnalysisModePill');

            if (modePill) {
                modePill.textContent = ga.mode === 'G' ? 'Mode G (Gemini Intelligence)' : 'Mode M (Scientometrics)';
            }
            if (execText) execText.textContent = ga.executive_summary || '';

            if (gapsList) {
                gapsList.innerHTML = '';
                (ga.novel_research_gaps || []).forEach((gap, idx) => {
                    const card = document.createElement('div');
                    card.className = 'topic-gap-card';
                    const titleStrong = document.createElement('strong');
                    titleStrong.textContent = `Gap ${idx + 1}: `;
                    card.appendChild(titleStrong);
                    card.appendChild(document.createTextNode(gap));
                    gapsList.appendChild(card);
                });
            }

            if (methodsList) {
                methodsList.innerHTML = '';
                (ga.recommended_methodology || []).forEach(m => {
                    const item = document.createElement('div');
                    item.className = 'topic-method-item';
                    const tick = document.createElement('span');
                    tick.style.color = 'var(--accent-forest)';
                    tick.style.fontWeight = '700';
                    tick.textContent = '✔';
                    const text = document.createElement('span');
                    text.textContent = m;
                    item.appendChild(tick);
                    item.appendChild(text);
                    methodsList.appendChild(item);
                });
            }

            if (pitfallsList) {
                pitfallsList.innerHTML = '';
                (ga.common_pitfalls || []).forEach(p => {
                    const item = document.createElement('div');
                    item.className = 'topic-method-item';
                    const cross = document.createElement('span');
                    cross.style.color = '#DC2626';
                    cross.style.fontWeight = '700';
                    cross.textContent = '✖';
                    const text = document.createElement('span');
                    text.textContent = p;
                    item.appendChild(cross);
                    item.appendChild(text);
                    pitfallsList.appendChild(item);
                });
            }

            // 6. Top Venues Table
            const venuesTbody = document.getElementById('topicVenuesTableBody');
            if (venuesTbody) {
                venuesTbody.innerHTML = '';
                (data.top_venues || []).forEach(v => {
                    const tr = document.createElement('tr');

                    const tdName = document.createElement('td');
                    tdName.style.fontWeight = '600';
                    tdName.textContent = v.venue_name;

                    const tdCount = document.createElement('td');
                    tdCount.textContent = (v.publication_count || 0).toLocaleString();

                    const tdQ = document.createElement('td');
                    const qSpan = document.createElement('span');
                    const qVal = v.quartile || 'Unranked';
                    qSpan.className = 'topic-venue-badge ' + (qVal === 'Q1' ? 'q1' : (qVal === 'Q2' ? 'q2' : (qVal === 'Q3' ? 'q3' : 'q4')));
                    qSpan.textContent = qVal;
                    tdQ.appendChild(qSpan);

                    const tdCore = document.createElement('td');
                    if (v.core_rank) {
                        const coreSpan = document.createElement('span');
                        coreSpan.className = 'topic-venue-badge core';
                        coreSpan.textContent = 'CORE ' + v.core_rank;
                        tdCore.appendChild(coreSpan);
                    } else {
                        tdCore.textContent = '--';
                        tdCore.style.color = 'var(--text-tertiary)';
                    }

                    const tdSjr = document.createElement('td');
                    tdSjr.textContent = v.sjr !== null && v.sjr !== undefined ? Number(v.sjr).toFixed(2) : '--';

                    const tdH = document.createElement('td');
                    tdH.textContent = v.h_index !== null && v.h_index !== undefined ? String(v.h_index) : '--';

                    tr.appendChild(tdName);
                    tr.appendChild(tdCount);
                    tr.appendChild(tdQ);
                    tr.appendChild(tdCore);
                    tr.appendChild(tdSjr);
                    tr.appendChild(tdH);
                    venuesTbody.appendChild(tr);
                });
            }

            // 7. Seminal & Recent Papers
            renderTopicPaperList('topicSeminalPapersContainer', data.seminal_papers || []);
            renderTopicPaperList('topicRecentPapersContainer', data.recent_papers || []);
        }

        function renderTopicPaperList(containerId, papers) {
            const container = document.getElementById(containerId);
            if (!container) return;
            container.innerHTML = '';

            if (papers.length === 0) {
                const emptyMsg = document.createElement('div');
                emptyMsg.style.fontSize = '0.8rem';
                emptyMsg.style.color = 'var(--text-tertiary)';
                emptyMsg.style.padding = '0.5rem';
                emptyMsg.textContent = 'No paper records identified for this subset.';
                container.appendChild(emptyMsg);
                return;
            }

            papers.forEach(p => {
                const card = document.createElement('div');
                card.style.background = '#FFFFFF';
                card.style.border = '1px solid var(--border-subtle)';
                card.style.borderRadius = '6px';
                card.style.padding = '0.75rem 0.9rem';

                const titleLink = document.createElement('a');
                titleLink.href = p.doi ? `https://doi.org/${p.doi}` : (p.openalex_url || '#');
                titleLink.target = '_blank';
                titleLink.rel = 'noopener noreferrer';
                titleLink.style.fontWeight = '600';
                titleLink.style.fontSize = '0.84rem';
                titleLink.style.color = 'var(--text-ink)';
                titleLink.style.textDecoration = 'none';
                titleLink.textContent = p.title;

                const metaDiv = document.createElement('div');
                metaDiv.style.display = 'flex';
                metaDiv.style.justifyContent = 'space-between';
                metaDiv.style.alignItems = 'center';
                metaDiv.style.marginTop = '0.4rem';
                metaDiv.style.fontSize = '0.74rem';
                metaDiv.style.color = 'var(--text-secondary)';

                const venueSpan = document.createElement('span');
                venueSpan.textContent = `${p.venue_name || 'Academic Venue'} (${p.publication_year || 'Year'})`;

                const citBadge = document.createElement('span');
                citBadge.style.background = 'var(--accent-forest-subtle)';
                citBadge.style.color = 'var(--accent-forest)';
                citBadge.style.padding = '2px 6px';
                citBadge.style.borderRadius = '4px';
                citBadge.style.fontWeight = '700';
                citBadge.textContent = `${(p.citations || 0).toLocaleString()} citations`;

                metaDiv.appendChild(venueSpan);
                metaDiv.appendChild(citBadge);

                card.appendChild(titleLink);
                card.appendChild(metaDiv);
                container.appendChild(card);
            });
        }
