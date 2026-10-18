// SciGraph 2.4 PRO - Predictor View Logic

        function loadSample(id) {
            const input = document.getElementById('paperInput');
            if (input) input.value = id;
            const presetSel = document.getElementById('presetPaperSelect');
            if (presetSel) {
                for (let i = 0; i < presetSel.options.length; i++) {
                    if (presetSel.options[i].value === id) {
                        presetSel.selectedIndex = i;
                        break;
                    }
                }
            }
            runPrediction();
        }

        function onPresetSelect(id) {
            if (!id) return;
            const input = document.getElementById('paperInput');
            if (input) input.value = id;
            runPrediction();
        }



        async function runPrediction() {
            const inputEl = document.getElementById('paperInput');
            const inputVal = inputEl ? inputEl.value.trim() : '';
            if (!inputVal) return;

            const empty = document.getElementById('predictionEmptyState');
            const area = document.getElementById('predictionResultsArea');
            const loader = document.getElementById('predictionLoadingArea');
            const loadTitle = document.getElementById('predictionLoadingTitle');
            const loadSub = document.getElementById('predictionLoadingSubtitle');

            if (empty) empty.style.display = 'none';
            if (area) area.style.display = 'none';
            if (loader) {
                loader.style.display = 'block';
                if (loadTitle) loadTitle.textContent = `Extracting 2-Hop Citation Graph & Evaluating ${inputVal}...`;
                if (loadSub) loadSub.innerHTML = 'Enforcing strict temporal boundary firewall ($T_{\\text{cutoff}} = \\text{pub\\_year} + 2$) and computing inference features...';
            }

            try {
                const res = await fetch(`/api/v1/papers/predict/${encodeURIComponent(inputVal)}?mode=${currentPredictionMode}`);
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    console.error('Prediction Error:', err);
                    if (loader) {
                        loader.innerHTML = `<div style="color: var(--destructive-red); font-weight: 700; padding: 1rem;">Prediction Error (HTTP ${res.status}): ${err.detail || 'Could not load paper'}</div>`;
                    }
                    return;
                }
                const data = await res.json();
                if (loader) loader.style.display = 'none';
                renderPrediction(data);
            } catch (err) {
                console.error('Network error connecting to API:', err);
                if (loader) {
                    loader.innerHTML = `<div style="color: var(--destructive-red); font-weight: 700; padding: 1rem;">Network Error: ${err.message}</div>`;
                }
            }
        }

        function renderTrajectoryBars(points, cutoffYear, pubYear) {
            const host = document.getElementById('predTrajectoryChart');
            if (!host) return;
            host.innerHTML = '';
            if (!points || !Array.isArray(points) || points.length === 0) {
                host.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-secondary); padding: 1rem;">No yearly citation trajectory cataloged for this work.</div>';
                return;
            }
            const sorted = [...points].sort((a, b) => a.year - b.year);
            const maxV = Math.max(1, ...sorted.map(p => Number(p.cited_by_count || 0)));
            sorted.forEach(p => {
                const count = Number(p.cited_by_count || 0);
                const yr = Number(p.year);
                const isCutoff = p.is_cutoff || (cutoffYear && yr === Number(cutoffYear));
                const isFuture = p.is_future_horizon || (cutoffYear && yr > Number(cutoffYear));

                const col = document.createElement('div');
                col.className = 'year-bar-col';
                col.title = `Year ${yr}: ${count.toLocaleString()} citations ${isCutoff ? '(T_cutoff Boundary)' : (isFuture ? '(Post-Cutoff Ground Truth)' : '(Pre-Cutoff Ingested)')}`;

                const val = document.createElement('div');
                val.className = 'year-bar-val';
                val.textContent = count > 9999 ? (count / 1000).toFixed(1) + 'k' : String(count);

                const bar = document.createElement('div');
                let cls = 'year-bar';
                if (isCutoff) cls += ' cutoff';
                else if (isFuture) cls += ' window';
                bar.className = cls;
                bar.style.height = Math.max(6, Math.round((count / maxV) * 90)) + 'px';

                const lab = document.createElement('div');
                lab.className = 'year-bar-label';
                lab.textContent = yr === Number(cutoffYear) ? `${yr} ★` : String(yr);
                if (yr === Number(cutoffYear)) {
                    lab.style.fontWeight = '700';
                    lab.style.color = '#B45309';
                }

                col.appendChild(val);
                col.appendChild(bar);
                col.appendChild(lab);
                host.appendChild(col);
            });
        }

        function renderPrediction(data) {
            const empty = document.getElementById('predictionEmptyState');
            if (empty) empty.style.display = 'none';
            const area = document.getElementById('predictionResultsArea');
            if (area) area.style.display = 'block';

            // Populate Lead Metadata
            safeSetText('resPaperTitle', data.title || 'Untitled');
            safeSetText('resPubYear', data.publication_year || '—');
            safeSetText('resCutoffYear', data.cutoff_year || (data.publication_year ? (data.publication_year + 2) : '—'));
            safeSetText('resCutoffCits', fmtNum(cutoffCount(data)));
            safeSetText('resSplitStatus', data.in_train_split ? 'Training Split (2012–2016)' : 'Held-Out Blind Test');

            // OpenAlex Link
            const cleanId = (data.paper_id || '').split('/').pop();
            const linkEl = document.getElementById('resOpenAlexLink');
            if (linkEl && cleanId) {
                linkEl.href = `https://openalex.org/${cleanId}`;
                linkEl.textContent = `OpenAlex Verified Record (${cleanId}) ↗`;
            }

            // Authors Line
            const authorsList = data.authors && data.authors.length > 0 ? data.authors.join(', ') : 'Authors cataloged in OpenAlex';
            safeSetText('resAuthorsLine', authorsList);

            // Venue Line & Prestige
            let venueText = 'Peer-Reviewed Literature';
            let venuePrestige = 'Standard Peer-Reviewed';
            if (data.venue && data.venue.venue_name) {
                const quartile = data.venue.quartile || data.venue.core_rank || 'Peer-Reviewed';
                venueText = `Published in <strong>${data.venue.venue_name}</strong> (${quartile})`;
                venuePrestige = `${data.venue.venue_name.slice(0, 36)} (${quartile})`;
            }
            safeSetHtml('resVenueLine', venueText);
            safeSetText('econVenuePrestige', venuePrestige);

            // Scientometrics & Access Economics Strip (Clean without zero placeholders)
            const econ = data.access_economics || {};
            if (econ.is_oa) {
                safeSetHtml('econModelVal', `<span style="color: var(--success-green); font-weight: 700;">Open Access (${(econ.oa_status || 'OA').toUpperCase()})</span>`);
                if (econ.apc_paid_usd != null && econ.apc_paid_usd > 0) {
                    safeSetText('econApcVal', `$${Math.round(econ.apc_paid_usd).toLocaleString()} USD (Author Paid)`);
                } else if (econ.apc_list_usd != null && econ.apc_list_usd > 0) {
                    safeSetText('econApcVal', `$${Math.round(econ.apc_list_usd).toLocaleString()} USD (List Price)`);
                } else {
                    safeSetText('econApcVal', `Institutional / Diamond OA (No APC required)`);
                }

                // Show OA Advantage ONLY if paper is Open Access and advantage is positive
                const advPct = econ.empirical_cohort_oa_advantage_pct;
                const advCol = document.getElementById('econAdvCol');
                if (advCol) {
                    if (advPct != null && advPct > 0) {
                        advCol.style.display = 'block';
                        safeSetText('econAdvVal', `+${advPct.toFixed(1)}% citation volume`);
                    } else {
                        advCol.style.display = 'none';
                    }
                }
            } else {
                safeSetHtml('econModelVal', `<span style="color: var(--text-ink); font-weight: 600;">Traditional Subscription (Paywalled)</span>`);
                safeSetText('econApcVal', `Subscription Access (No author APC required)`);
                const advCol = document.getElementById('econAdvCol');
                if (advCol) advCol.style.display = 'none';
            }

            // CD Disruption Index
            const lineage = data.lineage || {};
            if (lineage.cd_disruption_index != null) {
                const cd = Number(lineage.cd_disruption_index);
                const cdDesc = cd < -0.01 ? 'Consolidating milestone' : (cd > 0.01 ? 'Disruptive innovation' : 'Neutral lineage');
                safeSetText('econDisruptionVal', `${cd.toFixed(3)} (${cdDesc})`);
            } else {
                safeSetText('econDisruptionVal', `Cataloged Lineage`);
            }

            // Column 1: Model Prediction & Calibrated Probabilities
            const badgeEl = document.getElementById('resImpactBadge');
            const predCls = data.predicted_impact_class;
            if (badgeEl) {
                badgeEl.className = 'outcome-badge';
                if (predCls === 2) {
                    badgeEl.classList.add('high');
                    badgeEl.textContent = 'High Impact (≥90th)';
                } else if (predCls === 1) {
                    badgeEl.classList.add('medium');
                    badgeEl.textContent = 'Medium Impact (50–90th)';
                } else {
                    badgeEl.classList.add('low');
                    badgeEl.textContent = 'Low Impact (<50th)';
                }
            }

            const probs = data.calibrated_probabilities || data.class_probabilities || {};
            const pickProb = (keys) => {
                for (const k of keys) {
                    if (probs[k] != null) return (Number(probs[k]) * 100).toFixed(1);
                }
                return null;
            };
            const pHigh = pickProb(['High (>=90th)', 'High']);
            const pMed = pickProb(['Medium (50-90th)', 'Medium']);
            const pLow = pickProb(['Low (<50th)', 'Low']);

            safeSetText('probHighVal', pHigh != null ? `${pHigh}%` : '—');
            const bHigh = document.getElementById('probHighBar');
            if (bHigh) bHigh.style.width = pHigh != null ? `${pHigh}%` : '0%';

            safeSetText('probMedVal', pMed != null ? `${pMed}%` : '—');
            const bMed = document.getElementById('probMedBar');
            if (bMed) bMed.style.width = pMed != null ? `${pMed}%` : '0%';

            safeSetText('probLowVal', pLow != null ? `${pLow}%` : '—');
            const bLow = document.getElementById('probLowBar');
            if (bLow) bLow.style.width = pLow != null ? `${pLow}%` : '0%';

            const predWindow = data.predicted_future_citations_5y != null ? data.predicted_future_citations_5y : data.citations_in_window;
            safeSetText('resPredWindowCits', fmtNum(predWindow));

            // Column 2: Side-by-Side Ground Truth Empirical Verification
            const bt = data.backtest;
            const truthPanel = document.getElementById('sideBySideTruthPanel');
            if (truthPanel) {
                if (bt && bt.has_ground_truth) {
                    truthPanel.style.display = 'flex';
                    const actualCls = bt.actual_impact_class;
                    const actualBadge = document.getElementById('btActualBadge');
                    if (actualBadge) {
                        actualBadge.className = 'outcome-badge';
                        if (actualCls === 2) {
                            actualBadge.classList.add('high');
                            actualBadge.textContent = 'High Impact (≥90th)';
                        } else if (actualCls === 1) {
                            actualBadge.classList.add('medium');
                            actualBadge.textContent = 'Medium Impact (50–90th)';
                        } else {
                            actualBadge.classList.add('low');
                            actualBadge.textContent = 'Low Impact (<50th)';
                        }
                    }

                    const matchBadge = document.getElementById('btMatchBadge');
                    if (matchBadge) {
                        if (bt.is_correct_class) {
                            matchBadge.className = 'verdict-match-tag match';
                            matchBadge.innerHTML = '✓ Prediction Verified Correct';
                        } else {
                            matchBadge.className = 'verdict-match-tag misaligned';
                            matchBadge.innerHTML = 'Class Misaligned';
                        }
                    }

                    safeSetText('btCutoffStatVal', fmtNum(cutoffCount(data)));
                    safeSetText('btActualCitsVal', fmtNum(bt.actual_future_citations_5y != null ? bt.actual_future_citations_5y : bt.actual_window));
                    safeSetText('btPredCitsVal', fmtNum(data.predicted_future_citations_5y));
                    
                    const pctVal = bt.cohort_percentile != null ? Number(bt.cohort_percentile).toFixed(1) : null;
                    safeSetText('btCohortPctVal', pctVal != null ? `${pctVal}th percentile` : 'Evaluated against cohort');
                    
                    const startY = (data.publication_year || 0) + 3;
                    const endY = (data.publication_year || 0) + 5;
                    safeSetText('btWindowRangeVal', `${startY}–${endY} (5Y Closed)`);
                    safeSetText('btProvenanceTag', bt.provenance || 'OpenAlex (works.parquet)');
                } else {
                    truthPanel.style.display = 'none';
                }
            }

            // Timeline bar chart (Teacher Proof)
            renderTrajectoryBars(data.trajectory, data.cutoff_year, data.publication_year);

            // Update Provenance Pill for Mode M vs Mode G
            const provPill = document.getElementById('forecastProvenancePill');
            if (provPill) {
                if (data.mode === 'G' || data.gemini_reasoning) {
                    provPill.textContent = 'Gemini 2.5 Flash (Mode G)';
                    provPill.style.background = '#DBEAFE';
                    provPill.style.color = '#1D4ED8';
                } else {
                    provPill.textContent = 'M1_LogisticRegression (Mode M)';
                    provPill.style.background = '';
                    provPill.style.color = '';
                }
            }

            // Update Gemini Reasoning Card
            const geminiCard = document.getElementById('geminiReasoningCard');
            const geminiText = document.getElementById('geminiReasoningText');
            const geminiConf = document.getElementById('geminiConfidenceScore');
            if (geminiCard && geminiText) {
                if (data.gemini_reasoning) {
                    geminiCard.style.display = 'block';
                    geminiText.textContent = data.gemini_reasoning;
                    if (geminiConf && data.confidence) {
                        geminiConf.textContent = `Confidence: ${(data.confidence * 100).toFixed(0)}%`;
                    }
                } else {
                    geminiCard.style.display = 'none';
                }
            }

            // Render Subgraph
            renderSubgraph(data);
        }

