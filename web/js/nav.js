// SciGraph 2.4 PRO - Navigation & Mode Synchronization

let currentTopicMode = 'G';
let currentPredictionMode = 'M';

        function switchTab(viewId) {
            document.querySelectorAll('.view-pane').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

            const targetView = document.getElementById(viewId);
            if (targetView) targetView.classList.add('active');

            const idMap = {
                'viewTeacherVerify': { link: 'tabLinkTeacherVerify', title: 'Verify Our Work' },
                'viewPredictor': { link: 'tabLinkPredictor', title: 'Citation Impact Predictor' },
                'viewTopicFeasibility': { link: 'tabLinkTopicFeasibility', title: 'Topic Feasibility & Success' },
                'viewVerifyLab': { link: 'tabLinkVerify', title: 'Verification Lab' },
                'viewAudit': { link: 'tabLinkAudit', title: 'Temporal & Leakage Audit' },
                'viewBenchmarks': { link: 'tabLinkBenchmarks', title: 'Model Benchmarks' },
                'viewCompare': { link: 'tabLinkCompare', title: 'Paper Comparison' }
            };

            if (idMap[viewId]) {
                const navLink = document.getElementById(idMap[viewId].link);
                if (navLink) navLink.classList.add('active');
                safeSetText('activeBreadcrumb', idMap[viewId].title);
            }

            // Lazy load data for tabs when selected
            if (viewId === 'viewTeacherVerify') {
                if (typeof teacherCorpusSamples !== 'undefined' && teacherCorpusSamples.length === 0) loadTeacherCorpus();
                loadTeacherScoreboardAndFooter();
            }
            if (viewId === 'viewTopicFeasibility') {
                loadTopicPresets();
            }
            if (viewId === 'viewVerifyLab') loadVerifySummary();
            if (viewId === 'viewAudit') loadAuditResults();
            if (viewId === 'viewBenchmarks') loadBenchmarks();
        }


        function setTopicMode(mode) {
            currentTopicMode = mode;
            const btnM = document.getElementById('topicModeBtnM');
            const btnG = document.getElementById('topicModeBtnG');
            if (btnM) btnM.classList.toggle('active', mode === 'M');
            if (btnG) btnG.classList.toggle('active', mode === 'G');

            // Sync with global mode toggle
            if (typeof setPredictionMode === 'function') {
                const topM = document.getElementById('topbarModeBtnM');
                const topG = document.getElementById('topbarModeBtnG');
                if (topM && topG) {
                    topM.classList.toggle('active', mode === 'M');
                    topG.classList.toggle('active', mode === 'G');
                }
            }

            // Update pill if results are visible
            const pill = document.getElementById('topicAnalysisModePill');
            if (pill) {
                pill.textContent = mode === 'G' ? 'Mode G (Gemini Intelligence)' : 'Mode M (Scientometrics)';
                pill.style.background = mode === 'G' ? '#DBEAFE' : '#E5E7EB';
                pill.style.color = mode === 'G' ? '#1E40AF' : '#374151';
            }
        }


        function setPredictionMode(mode) {
            currentPredictionMode = (mode || 'M').toUpperCase();
            const isM = (currentPredictionMode === 'M');
            
            // Global Synchronization: Sync ALL mode buttons and variables
            currentTopicMode = currentPredictionMode;
            const syncBtn = (idM, idG) => {
                const bM = document.getElementById(idM);
                const bG = document.getElementById(idG);
                if (bM) bM.classList.toggle('active', isM);
                if (bG) bG.classList.toggle('active', !isM);
            };
            syncBtn('topbarModeBtnM', 'topbarModeBtnG');
            syncBtn('predModeBtnM', 'predModeBtnG');
            syncBtn('teacherModeBtnM', 'teacherModeBtnG');
            syncBtn('topicModeBtnM', 'topicModeBtnG');
            
            const topicPill = document.getElementById('topicActiveEnginePill');
            if (topicPill) {
                topicPill.textContent = isM ? 'Mode M (Scientometrics)' : 'Mode G (Gemini Intelligence)';
                topicPill.style.background = isM ? '#F1F5F9' : '#DBEAFE';
                topicPill.style.color = isM ? '#475569' : '#1E40AF';
            }

            // If a prediction or teacher view is currently displayed, re-run with selected mode
            const activeView = document.querySelector('.view-pane.active');
            if (activeView && activeView.id === 'viewPredictor') {
                const resultsArea = document.getElementById('predictionResultsArea');
                if (resultsArea && resultsArea.style.display !== 'none') {
                    runPrediction();
                }
            } else if (activeView && activeView.id === 'viewTeacherVerify') {
                const teachRes = document.getElementById('teacherFreezeResult');
                if (teachRes && teachRes.style.display !== 'none') {
                    teacherPredictAndCompare();
                }
            }
        }

