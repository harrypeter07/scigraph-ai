// Published measurements are distinct from the live server's loaded model.
let forecastEvidenceRequest = null;

const evidencePercent = value => typeof value === 'number' && Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : '—';
const evidenceNumber = value => typeof value === 'number' ? value.toLocaleString('en-US', {maximumFractionDigits: 0}) : '—';
const evidenceF1 = value => typeof value === 'number' ? value.toFixed(3) : '—';

function evidenceSet(key, value) {
    document.querySelectorAll(`[data-evidence="${key}"]`).forEach(node => { node.textContent = value; });
}

function evidenceCell(row, value) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.appendChild(cell);
    return cell;
}

function renderForecastEvidence(data) {
    const primary = data.primary;
    if (!primary || typeof primary.accuracy !== 'number' || !primary.n) throw new Error('Evaluation report is incomplete.');
    const ci = Array.isArray(primary.accuracy_ci95) ? primary.accuracy_ci95.map(evidencePercent).join(' – ') : 'Unavailable';
    const values = {
        accuracy: evidencePercent(primary.accuracy), 'test-n': evidenceNumber(primary.n),
        'macro-f1': evidenceF1(primary.macro_f1), 'high-precision': evidencePercent(primary.per_class?.['2']?.precision),
        'gnn-accuracy': evidencePercent(data.gnn?.accuracy), ci, date: data.evaluated_date,
        'baseline-accuracy': evidencePercent(data.baseline?.accuracy), 'pooled-accuracy': evidencePercent(data.retrospective_pooled?.accuracy),
        'corpus-n': evidenceNumber(data.dataset?.total), 'train-n': evidenceNumber(data.dataset?.train),
        'selection-n': evidenceNumber(data.dataset?.selection), 'calibration-n': evidenceNumber(data.dataset?.calibration)
    };
    Object.entries(values).forEach(([key, value]) => evidenceSet(key, value));
    safeSetText('topbarHeldoutAcc', evidencePercent(primary.accuracy));
    const servingVerified = data.serving?.verified === true;
    const runtimeText = servingVerified ? 'Evaluated bundle verified' : 'Runtime not verified';
    ['evidenceRuntime', 'sidebarRuntime'].forEach(id => {
        const node = document.getElementById(id);
        if (node) { node.textContent = runtimeText; node.classList.toggle('verified', servingVerified); }
    });
    safeSetText('sidebarCorpus', `${evidenceNumber(data.dataset?.total)} evaluated works`);
    safeSetText('sidebarModel', servingVerified ? data.model_display_name : 'Published evaluation');
    safeSetText('sidebarEvaluation', `2020 test · ${evidencePercent(primary.accuracy)}`);
    const models = document.getElementById('evidenceModelRows');
    if (models) {
        models.replaceChildren();
        [
            ['Gradient boosting', 'Selected forecast', primary.accuracy, primary.macro_f1, true],
            ['Citation GraphSAGE', 'Experimental · same cohort', data.gnn?.accuracy, data.gnn?.macro_f1, false],
            ['Majority class', 'Always predicts Low', data.baseline?.accuracy, data.baseline?.macro_f1, false]
        ].forEach(([name, role, accuracy, f1, selected]) => {
            const row = document.createElement('tr');
            if (selected) row.className = 'selected';
            const title = evidenceCell(row, name);
            const note = document.createElement('span'); note.className = 'model-role'; note.textContent = role; title.appendChild(note);
            evidenceCell(row, evidencePercent(accuracy)); evidenceCell(row, evidenceF1(f1)); models.appendChild(row);
        });
    }
    safeSetText('evidenceGnnRole', `Validation retained paper features for the selected forecast. The experimental GNN uses real citation messages; its expanded 2018–2020 test graph covers ${evidencePercent(data.gnn?.coverage_pooled)} of works. Graph coverage is different from prediction accuracy.`);
    const fields = document.getElementById('evidenceFieldRows');
    if (fields) {
        fields.replaceChildren();
        Object.entries(primary.by_field || {}).forEach(([name, result]) => {
            const item = document.createElement('div');
            const label = document.createElement('div'); label.className = 'field-label';
            const title = document.createElement('span'); title.textContent = name;
            const score = document.createElement('strong'); score.textContent = evidencePercent(result.accuracy);
            label.append(title, score);
            const track = document.createElement('div'); track.className = 'field-track';
            const bar = document.createElement('div'); bar.className = 'field-fill'; bar.style.width = `${Math.min(100, Math.max(0, result.accuracy*100))}%`;
            track.appendChild(bar); item.append(label, track); fields.appendChild(item);
        });
    }
    const classes = document.getElementById('evidenceClassRows');
    if (classes) {
        classes.replaceChildren();
        ['Low', 'Medium', 'High'].forEach((name, index) => {
            const stats = primary.per_class?.[String(index)] || {};
            const row = document.createElement('tr');
            [name, evidencePercent(stats.precision), evidencePercent(stats.recall), evidenceNumber(stats.support)].forEach(value => evidenceCell(row, value));
            classes.appendChild(row);
        });
    }
    const limitations = document.getElementById('evidenceLimitations');
    if (limitations) {
        limitations.replaceChildren();
        (data.limitations || []).forEach(text => { const li = document.createElement('li'); li.textContent = text; limitations.appendChild(li); });
    }
    loadVerifiedExamplePresets(data);
    const examples = document.getElementById('forecastExamples');
    if (examples) {
        examples.replaceChildren();
        (data.examples || []).forEach(example => {
            const button = document.createElement('button'); button.type = 'button'; button.className = 'research-example';
            const content = document.createElement('div');
            const title = document.createElement('strong'); title.textContent = example.title;
            const subtitle = document.createElement('small'); subtitle.textContent = `${example.publication_year} · ${example.primary_field_name}`;
            content.append(title, subtitle);
            const arrow = document.createElement('span'); arrow.className = 'example-arrow'; arrow.textContent = '↗'; arrow.setAttribute('aria-hidden', 'true');
            button.append(content, arrow); button.addEventListener('click', () => loadSample(example.work_id.split('/').pop()));
            examples.appendChild(button);
        });
    }
    // The verification view must not overwrite the current headline with legacy metrics.
    safeSetText('teacherScoreModelAcc', evidencePercent(primary.accuracy));
    safeSetText('teacherScoreModelF1', evidenceF1(primary.macro_f1));
    safeSetText('teacherScoreMajAcc', evidencePercent(data.baseline?.accuracy));
    safeSetText('teacherScoreMajF1', evidenceF1(data.baseline?.macro_f1));
    safeSetText('teacherScoreEarlyAcc', 'Not evaluated');
    safeSetText('teacherScoreEarlyF1', '—');
    safeSetText('evidenceLoadError', servingVerified ? '' : 'These are published evaluation results. The current server’s model bundle could not be verified.');
}

async function loadVerifiedExamplePresets(evidence) {
    const select = document.getElementById('presetPaperSelect');
    if (!select) return;
    for (const path of ['/data/verified_examples.json', '/static/data/verified_examples.json']) {
        try {
            const response = await fetch(path, {cache:'no-store'});
            if (!response.ok) continue;
            const manifest = await response.json();
            if (manifest.forecast_bundle_sha256 !== evidence.forecast_bundle_sha256) continue;
            select.replaceChildren();
            const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a verified correct example (2020)…'; select.appendChild(placeholder);
            for (const example of manifest.examples || []) {
                if (example.verified_correct !== true) continue;
                const option = document.createElement('option');
                option.value = example.paper_id;
                option.textContent = `${example.predicted_label} · ${example.field} · ${example.title}`;
                select.appendChild(option);
            }
            select.disabled = select.options.length < 2;
            return;
        } catch (_) { /* Try the other supported static mount. */ }
    }
    select.replaceChildren();
    const option = document.createElement('option'); option.textContent = 'Verified examples unavailable on this server'; select.appendChild(option); select.disabled = true;
}

async function loadForecastEvidence() {
    if (!forecastEvidenceRequest) {
        forecastEvidenceRequest = (async () => {
            for (const url of ['/api/v1/papers/forecast-evidence', '/static/data/forecast_evidence.json', '/data/forecast_evidence.json']) {
                try {
                    const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(6000)});
                    if (!response.ok) continue;
                    const data = await response.json();
                    if (!data.primary || typeof data.primary.accuracy !== 'number') continue;
                    if (url.startsWith('/data/')) {
                        const link = document.getElementById('evidenceDownload'); if (link) link.href = url;
                    }
                    renderForecastEvidence(data);
                    return data;
                } catch (error) { /* Try the published snapshot without claiming runtime verification. */ }
            }
            safeSetText('evidenceLoadError', 'Evaluation evidence could not be loaded. No accuracy claim is displayed.');
            safeSetText('sidebarRuntime', 'Evidence unavailable');
            safeSetText('forecastExamples', 'Examples are unavailable. You can still enter a paper identifier above.');
            return null;
        })();
    }
    return forecastEvidenceRequest;
}
