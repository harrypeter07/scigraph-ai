// SciGraph 2.4 PRO - Teacher Verify & Provenance Reality Verification

        // ========================================================
        let teacherCorpusSamples = [];
        let teacherSelectedPaper = null;
        let teacherCurrentReceipt = null;

        async function loadTeacherCorpus() {
            const select = document.getElementById('teacherPaperSelect');
            const errBox = document.getElementById('teacherPaperError');
            if (!select) return;
            try {
                const resp = await fetch('/api/v1/papers/corpus-sample?limit=50&randomize=true');
                if (!resp.ok) throw new Error('Failed to load sample papers: HTTP ' + resp.status);
                const data = await resp.json();
                teacherCorpusSamples = data.sample || [];
                select.innerHTML = '';
                if (teacherCorpusSamples.length === 0) {
                    const opt = document.createElement('option');
                    opt.textContent = 'No papers available';
                    select.appendChild(opt);
                    return;
                }
                teacherCorpusSamples.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.paper_id;
                    opt.textContent = `${p.title} (${p.publication_year}, ${p.field})`;
                    select.appendChild(opt);
                });
                teacherSelectedPaper = teacherCorpusSamples[0];
                renderTeacherPaperCard(teacherSelectedPaper);
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = 'Error loading sample papers: ' + e.message;
                }
            }
        }


        function renderTeacherPaperCard(paper) {
            if (!paper) return;
            const card = document.getElementById('teacherPaperCard');
            const titleEl = document.getElementById('teacherPaperTitle');
            const idEl = document.getElementById('teacherPaperId');
            const yrEl = document.getElementById('teacherPaperYear');
            const fldEl = document.getElementById('teacherPaperField');
            const venEl = document.getElementById('teacherPaperVenue');

            if (card) card.style.display = 'block';
            if (titleEl) titleEl.textContent = paper.title || 'Untitled';
            if (idEl) idEl.textContent = paper.paper_id || '';
            if (yrEl) yrEl.textContent = String(paper.publication_year || '');
            if (fldEl) fldEl.textContent = paper.field || 'General';
            if (venEl) venEl.textContent = paper.venue_name || 'Not indexed';

            const frEl = document.getElementById('teacherFreezeResult');
            const rvEl = document.getElementById('teacherRevealResult');
            if (frEl) frEl.style.display = 'none';
            if (rvEl) rvEl.style.display = 'none';
        }

        function onTeacherPaperSelected(wid) {
            teacherSelectedPaper = teacherCorpusSamples.find(p => p.paper_id === wid) || null;
            renderTeacherPaperCard(teacherSelectedPaper);
        }

        function teacherPickRandomPaper() {
            if (teacherCorpusSamples.length === 0) return;
            const idx = Math.floor(Math.random() * teacherCorpusSamples.length);
            teacherSelectedPaper = teacherCorpusSamples[idx];
            const select = document.getElementById('teacherPaperSelect');
            if (select) select.value = teacherSelectedPaper.paper_id;
            renderTeacherPaperCard(teacherSelectedPaper);
        }

        function teacherPickPreset(idx) {
            if (teacherCorpusSamples.length === 0) return;
            const paper = teacherCorpusSamples[Math.min(idx, teacherCorpusSamples.length - 1)];
            teacherSelectedPaper = paper;
            const select = document.getElementById('teacherPaperSelect');
            if (select) select.value = paper.paper_id;
            renderTeacherPaperCard(paper);
        }

        async function teacherLoadPaperById() {
            const input = document.getElementById('teacherPaperIdInput');
            const raw = (input && input.value || '').trim();
            const errBox = document.getElementById('teacherPaperError');
            if (!raw) return;
            if (errBox) errBox.style.display = 'none';
            try {
                const resp = await fetch(`/api/v1/papers/predict/${encodeURIComponent(raw)}`);
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok) throw new Error(data.detail || ('Could not load paper HTTP ' + resp.status));
                teacherSelectedPaper = {
                    paper_id: (data.paper_id || raw).toString().split('/').pop(),
                    title: data.title,
                    publication_year: data.publication_year,
                    field: data.primary_field,
                    venue_name: (data.venue && data.venue.venue_name) || ''
                };
                const select = document.getElementById('teacherPaperSelect');
                if (select && ![...select.options].some(o => o.value === teacherSelectedPaper.paper_id)) {
                    const opt = document.createElement('option');
                    opt.value = teacherSelectedPaper.paper_id;
                    opt.textContent = `${teacherSelectedPaper.title} (${teacherSelectedPaper.publication_year}, ${teacherSelectedPaper.field || 'General'})`;
                    select.appendChild(opt);
                    select.value = teacherSelectedPaper.paper_id;
                } else if (select) {
                    select.value = teacherSelectedPaper.paper_id;
                }
                renderTeacherPaperCard(teacherSelectedPaper);
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = e.message;
                }
            }
        }

        function renderYearChart(early, windowRows, cutoffYear) {
            const host = document.getElementById('teacherYearChart');
            if (!host) return;
            host.innerHTML = '';
            const points = [];
            (early || []).forEach(p => points.push({ year: Number(p.year), count: Number(p.cited_by_count || 0), kind: Number(p.year) === Number(cutoffYear) ? 'cutoff' : 'early' }));
            (windowRows || []).forEach(p => points.push({ year: Number(p.year), count: Number(p.cited_by_count || 0), kind: 'window' }));
            points.sort((a, b) => a.year - b.year);
            const maxV = Math.max(1, ...points.map(p => p.count));
            points.forEach(p => {
                const col = document.createElement('div');
                col.className = 'year-bar-col';
                const val = document.createElement('div');
                val.className = 'year-bar-val';
                val.textContent = String(p.count);
                const bar = document.createElement('div');
                bar.className = 'year-bar ' + (p.kind === 'window' ? 'window' : (p.kind === 'cutoff' ? 'cutoff' : ''));
                bar.style.height = Math.max(4, Math.round((p.count / maxV) * 120)) + 'px';
                const lab = document.createElement('div');
                lab.className = 'year-bar-label';
                lab.textContent = String(p.year);
                col.appendChild(val);
                col.appendChild(bar);
                col.appendChild(lab);
                host.appendChild(col);
            });
            if (points.length === 0) {
                host.textContent = 'No yearly counts returned for this paper.';
            }
        }

        function renderCompareBars(predCits, actualCits) {
            const host = document.getElementById('teacherCompareBars');
            if (!host) return;
            host.innerHTML = '';
            const pred = Number(predCits || 0);
            const actual = Number(actualCits || 0);
            const maxV = Math.max(1, pred, actual);
            [
                { label: 'Predicted', count: pred, cls: '' },
                { label: 'Actual (OpenAlex)', count: actual, cls: 'window' }
            ].forEach(p => {
                const col = document.createElement('div');
                col.className = 'year-bar-col';
                const val = document.createElement('div');
                val.className = 'year-bar-val';
                val.textContent = fmtNum(p.count);
                const bar = document.createElement('div');
                bar.className = 'year-bar ' + p.cls;
                bar.style.height = Math.max(8, Math.round((p.count / maxV) * 90)) + 'px';
                const lab = document.createElement('div');
                lab.className = 'year-bar-label';
                lab.textContent = p.label;
                col.appendChild(val);
                col.appendChild(bar);
                col.appendChild(lab);
                host.appendChild(col);
            });
        }

        async function teacherPredictAndCompare() {
            if (!teacherSelectedPaper) return;
            const wid = teacherSelectedPaper.paper_id;
            const errBox = document.getElementById('teacherFreezeError');
            const resBox = document.getElementById('teacherFreezeResult');
            const loader = document.getElementById('teacherRunLoader');
            const btn = document.getElementById('btnTeacherFreeze');
            if (errBox) errBox.style.display = 'none';
            if (loader) loader.classList.add('on');
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Running...';
            }

            try {
                const resp = await fetch(`/api/v1/verify/compare/${encodeURIComponent(wid)}?mode=${currentPredictionMode}`);
                const payload = await resp.json().catch(() => ({}));
                if (!resp.ok) throw new Error(payload.detail || ('Live prediction failed: HTTP ' + resp.status));

                const pred = payload.prediction || {};
                const receipt = payload.receipt || {};
                const reveal = payload.reveal || {};
                const side = payload.side_by_side || {};
                teacherCurrentReceipt = receipt;

                if (resBox) {
                    resBox.style.display = 'block';
                    resBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }

                // Render Gemini reasoning if in Mode G
                const geminiTile = document.getElementById('teacherGeminiReasoningTile');
                const geminiTxt = document.getElementById('teacherGeminiReasoningText');
                if (geminiTile && geminiTxt) {
                    const gReason = side.gemini_reasoning || payload.gemini_reasoning || (pred && pred.gemini_reasoning);
                    if (gReason) {
                        geminiTile.style.display = 'block';
                        geminiTxt.textContent = gReason;
                    } else {
                        geminiTile.style.display = 'none';
                    }
                }

                const cutoffYear = pred.cutoff_year || receipt.cutoff_year;
                const earlyCits = side.citations_at_cutoff != null ? side.citations_at_cutoff : cutoffCount(pred);
                document.getElementById('teacherCutoffYear').textContent = String(cutoffYear || '—');
                document.getElementById('teacherEarlyCitations').textContent = fmtNum(earlyCits);

                const tableBody = document.getElementById('teacherEarlyCitationsTable');
                tableBody.innerHTML = '';
                const traj = side.early_trajectory || (receipt.frozen_inputs || {}).historical_citation_trajectory_to_cutoff || [];
                if (traj.length === 0) {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `<td>Up to ${cutoffYear}</td><td>${fmtNum(earlyCits)}</td><td>Pre-cutoff observed</td>`;
                    tableBody.appendChild(tr);
                } else {
                    traj.forEach(item => {
                        const tr = document.createElement('tr');
                        const status = item.year === cutoffYear ? 'Cutoff boundary' : 'Pre-cutoff observed';
                        tr.innerHTML = `<td>${item.year}</td><td>${fmtNum(item.cited_by_count)}</td><td>${status}</td>`;
                        tableBody.appendChild(tr);
                    });
                }

                const predCits = side.predicted_window_citations != null ? side.predicted_window_citations : pred.predicted_future_citations_5y;
                const actualCits = side.actual_window_citations != null ? side.actual_window_citations : reveal.window_total_citations;
                document.getElementById('teacherPredWindowCits').textContent = fmtNum(predCits);
                document.getElementById('teacherActualWindowCits').textContent = fmtNum(actualCits);

                setTierBadge(document.getElementById('teacherPredictedTier'), side.predicted_tier || pred.predicted_impact_label, side.predicted_class);
                setTierBadge(document.getElementById('teacherActualTierInline'), side.actual_tier || reveal.actual_tier, side.actual_class);

                const matchEl = document.getElementById('teacherCmpMatch');
                if (side.is_correct) {
                    matchEl.className = 'verdict-match-tag match';
                    matchEl.textContent = 'Correct';
                } else {
                    matchEl.className = 'verdict-match-tag misaligned';
                    matchEl.textContent = 'Misaligned';
                }

                const probs = side.probabilities || pred.calibrated_probabilities || {};
                const probContainer = document.getElementById('teacherProbabilities');
                probContainer.innerHTML = '';
                Object.entries(probs).forEach(([k, v]) => {
                    if (v == null) return;
                    const span = document.createElement('span');
                    span.style.background = 'var(--bg-surface-subtle)';
                    span.style.padding = '3px 8px';
                    span.style.borderRadius = '4px';
                    span.textContent = `${k}: ${(Number(v) * 100).toFixed(1)}%`;
                    probContainer.appendChild(span);
                });

                const th = side.cohort_thresholds || reveal.cohort_thresholds || {};
                document.getElementById('teacherCohortInline').textContent = `${fmtNum(th.p50_threshold)} / ${fmtNum(th.p90_threshold)}`;
                document.getElementById('teacherReceiptHash').textContent = side.receipt_sha256 || receipt.receipt_sha256 || '—';

                const yearly = side.yearly_actual || reveal.yearly_citations_breakdown || [];
                renderYearChart(traj, yearly, cutoffYear);
                renderCompareBars(predCits, actualCits);
                if (pred && document.getElementById('cyTeacher')) {
                    renderSubgraph(pred, 'cyTeacher');
                }

                const rvEl = document.getElementById('teacherRevealResult');
                if (rvEl) rvEl.style.display = 'block';
                const tbody = document.getElementById('teacherRealityTable');
                if (tbody) {
                    tbody.innerHTML = '';
                    yearly.forEach(item => {
                        const tr = document.createElement('tr');
                        tr.innerHTML = `<td>${item.year}</td><td>${fmtNum(item.cited_by_count)}</td><td>Post-cutoff outcome</td>`;
                        tbody.appendChild(tr);
                    });
                }
                const totalEl = document.getElementById('teacherRealityTotal');
                if (totalEl) totalEl.textContent = fmtNum(actualCits);
                const p50El = document.getElementById('teacherCohortP50');
                const p90El = document.getElementById('teacherCohortP90');
                if (p50El) p50El.textContent = fmtNum(th.p50_threshold);
                if (p90El) p90El.textContent = fmtNum(th.p90_threshold);
                const actualTierEl = document.getElementById('teacherActualTier');
                if (actualTierEl) actualTierEl.textContent = side.actual_tier || '—';
                const badge = document.getElementById('teacherVerdictBadge');
                if (badge) {
                    badge.textContent = side.is_correct ? 'Correct' : 'Wrong';
                    badge.style.background = side.is_correct ? 'var(--success-bg)' : 'var(--destructive-bg)';
                    badge.style.color = side.is_correct ? 'var(--success-green)' : 'var(--destructive-red)';
                }
                const link = document.getElementById('teacherOpenAlexLink');
                if (link) link.href = `https://api.openalex.org/works/${wid}?select=id,publication_year,counts_by_year`;
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = e.message;
                }
            } finally {
                if (loader) loader.classList.remove('on');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Run prediction';
                }
            }
        }

        async function teacherFreezeAndPredict() {
            return teacherPredictAndCompare();
        }

        async function teacherRevealReality() {
            if (!teacherSelectedPaper) return;
            const wid = teacherSelectedPaper.paper_id;
            const errBox = document.getElementById('teacherRevealError');
            const resBox = document.getElementById('teacherRevealResult');
            if (errBox) errBox.style.display = 'none';

            try {
                const resp = await fetch(`/api/v1/verify/reveal/${wid}`);
                if (!resp.ok) throw new Error('Failed to reveal reality: HTTP ' + resp.status);
                const data = await resp.json();

                if (resBox) resBox.style.display = 'block';

                const tbody = document.getElementById('teacherRealityTable');
                tbody.innerHTML = '';
                (data.yearly_citations_breakdown || []).forEach(item => {
                    const tr = document.createElement('tr');
                    const td1 = document.createElement('td');
                    td1.textContent = String(item.year);
                    const td2 = document.createElement('td');
                    td2.textContent = String(item.cited_by_count);
                    const td3 = document.createElement('td');
                    const pubYr = (data.cohort_thresholds && data.cohort_thresholds.pub_year) || 2018;
                    td3.textContent = 'Post-cutoff outcome (Year ' + (item.year - pubYr) + ' after publication)';
                    tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
                    tbody.appendChild(tr);
                });

                document.getElementById('teacherRealityTotal').textContent = String(data.window_total_citations);
                document.getElementById('teacherCohortP50').textContent = String(data.cohort_thresholds.p50_threshold);
                document.getElementById('teacherCohortP90').textContent = String(data.cohort_thresholds.p90_threshold);
                document.getElementById('teacherActualTier').textContent = data.actual_tier;

                const badge = document.getElementById('teacherVerdictBadge');
                if (data.is_correct) {
                    badge.textContent = 'Correct';
                    badge.style.background = 'var(--success-bg)';
                    badge.style.color = 'var(--success-green)';
                } else {
                    badge.textContent = 'Wrong';
                    badge.style.background = 'var(--destructive-bg)';
                    badge.style.color = 'var(--destructive-red)';
                }

                // External Link to OpenAlex
                const link = document.getElementById('teacherOpenAlexLink');
                if (link) {
                    link.href = `https://api.openalex.org/works/${wid}?select=id,publication_year,counts_by_year`;
                }

                // Check receipt integrity
                const hash = data.receipt_sha256;
                if (hash) {
                    try {
                        const cResp = await fetch(`/api/v1/verify/receipt/${hash}/check`);
                        const cData = await cResp.json();
                        const integrityEl = document.getElementById('teacherReceiptIntegrity');
                        if (cData.valid) {
                            integrityEl.textContent = '✔ Receipt unchanged: Verified against immutable prediction ledger';
                            integrityEl.style.color = 'var(--success-green)';
                        } else {
                            integrityEl.textContent = '✖ Integrity check failed: ' + (cData.reason || 'Hash mismatch');
                            integrityEl.style.color = 'var(--destructive-red)';
                        }
                    } catch (ignore) {}
                }
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = e.message;
                }
            }
        }

        async function teacherRunSpotcheck() {
            const btn = document.getElementById('btnTeacherSpotcheck');
            const errBox = document.getElementById('teacherSpotcheckError');
            const resBox = document.getElementById('teacherSpotcheckResult');
            const tbody = document.getElementById('teacherSpotcheckTableBody');
            if (errBox) errBox.style.display = 'none';
            if (btn) {
                btn.disabled = true;
                btn.textContent = 'Spot-checking live OpenAlex...';
            }

            try {
                const resp = await fetch('/api/v1/verify/spotcheck?n=20');
                if (!resp.ok) throw new Error('Live spotcheck failed: HTTP ' + resp.status);
                const data = await resp.json();

                if (resBox) resBox.style.display = 'block';
                document.getElementById('teacherMatchRate').textContent = data.match_rate_percent.toFixed(1) + '%';
                document.getElementById('teacherMatchCount').textContent = String(data.matches);
                document.getElementById('teacherValidCount').textContent = String(data.valid_count);

                tbody.innerHTML = '';
                (data.sample_results || []).forEach(r => {
                    const tr = document.createElement('tr');

                    const tdTitle = document.createElement('td');
                    const a = document.createElement('a');
                    a.href = r.openalex_api_url || `https://openalex.org/${r.work_id}`;
                    a.target = '_blank';
                    a.rel = 'noopener noreferrer';
                    a.textContent = r.title || r.work_id;
                    tdTitle.appendChild(a);

                    const tdYear = document.createElement('td');
                    tdYear.textContent = String(r.publication_year);

                    const tdStored = document.createElement('td');
                    tdStored.textContent = String(r.stored_window);

                    const tdLive = document.createElement('td');
                    tdLive.textContent = r.live_window !== null ? String(r.live_window) : 'Network Error';

                    const tdMatch = document.createElement('td');
                    const matchSpan = document.createElement('span');
                    matchSpan.textContent = r.tier_match ? 'Match' : 'Drift';
                    matchSpan.style.color = r.tier_match ? 'var(--success-green)' : 'var(--warning-amber)';
                    matchSpan.style.fontWeight = '600';
                    tdMatch.appendChild(matchSpan);

                    const tdStatus = document.createElement('td');
                    tdStatus.textContent = r.status;

                    tr.appendChild(tdTitle);
                    tr.appendChild(tdYear);
                    tr.appendChild(tdStored);
                    tr.appendChild(tdLive);
                    tr.appendChild(tdMatch);
                    tr.appendChild(tdStatus);
                    tbody.appendChild(tr);
                });
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = 'Spotcheck Error: ' + e.message;
                }
            } finally {
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = 'Spot-check 20 random papers live against OpenAlex';
                }
            }
        }

        async function teacherRunLeakage() {
            if (!teacherSelectedPaper) return;
            const wid = teacherSelectedPaper.paper_id;
            const errBox = document.getElementById('teacherLeakageError');
            const resBox = document.getElementById('teacherLeakageResult');
            if (errBox) errBox.style.display = 'none';

            try {
                const resp = await fetch(`/api/v1/verify/leakage-controls/${wid}`);
                if (!resp.ok) throw new Error('Leakage test failed: HTTP ' + resp.status);
                const data = await resp.json();

                if (resBox) resBox.style.display = 'block';
                const neg = data.negative_control;
                const pos = data.positive_control;

                document.getElementById('teacherNegDesc').textContent = neg.description;
                document.getElementById('teacherNegAdded').textContent = `+${neg.citations_added.toLocaleString()} citations in ${neg.target_year}`;
                document.getElementById('teacherNegDelta').textContent = `Δ = ${neg.max_probability_delta.toFixed(6)}`;
                document.getElementById('teacherNegVerdict').textContent = neg.verdict;

                document.getElementById('teacherPosDesc').textContent = pos.description;
                document.getElementById('teacherPosAdded').textContent = `+${pos.citations_added} citations in ${pos.target_year}`;
                document.getElementById('teacherPosDelta').textContent = `Δ = ${pos.max_probability_delta.toFixed(4)}`;
                document.getElementById('teacherPosVerdict').textContent = pos.verdict;
            } catch (e) {
                if (errBox) {
                    errBox.style.display = 'block';
                    errBox.textContent = e.message;
                }
            }
        }

        async function loadTeacherScoreboardAndFooter() {
            try {
                const bResp = await fetch('/api/v1/papers/benchmark');
                if (bResp.ok) {
                    const bData = await bResp.json();
                    const summary = bData.summary || {};
                    const m1 = summary.M1_LogisticRegression || {};
                    const m0 = summary.M0_Majority || {};
                    const b1 = summary.B1_EarlyCitationsOnly || {};

                    if (m1.micro_f1_mean !== undefined) {
                        const acc = (m1.micro_f1_mean * 100).toFixed(1) + '%';
                        safeSetText('teacherScoreModelAcc', acc);
                        safeSetText('teacherScoreModelF1', m1.macro_f1_mean.toFixed(4));
                        safeSetText('topbarHeldoutAcc', acc + ' Acc');
                    }
                    if (m0.micro_f1_mean !== undefined) {
                        safeSetText('teacherScoreMajAcc', (m0.micro_f1_mean * 100).toFixed(1) + '%');
                        safeSetText('teacherScoreMajF1', m0.macro_f1_mean.toFixed(4));
                    }
                    if (b1.micro_f1_mean !== undefined) {
                        safeSetText('teacherScoreEarlyAcc', (b1.micro_f1_mean * 100).toFixed(1) + '%');
                        safeSetText('teacherScoreEarlyF1', b1.macro_f1_mean.toFixed(4));
                    }
                }
            } catch (ignore) {}

            try {
                const hResp = await fetch('/health');
                if (hResp.ok) {
                    const hData = await hResp.json();
                    const sn = (hData.data_snapshot && hData.data_snapshot.openalex_retrieved_utc) || '2026-10-15';
                    safeSetText('teacherFooterSnapshot', sn);
                    safeSetText('teacherFooterVersion', `${hData.service || 'SciGraph AI 2.0'} (${hData.model_name || 'M1_LogisticRegression'})`);
                }
            } catch (ignore) {}
        }
