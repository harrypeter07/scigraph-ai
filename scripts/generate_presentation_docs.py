"""Script to generate the updated DOCX file containing the comprehensive 4 suggestions and all details."""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def create_docx(filename="SciGraph_Presentation_Guide_Simple.docx"):
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(30, 41, 59)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("SciGraph AI 2.0: Complete Student Master Guide & Defense Companion")
    run_title.font.name = 'Arial'
    run_title.font.size = Pt(21)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(14, 116, 144)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("12th-Standard Plain English Guide: Every Abbreviation, The Teacher's 4 Big Suggestions & Presentation Script")
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_paragraph()

    # SECTION 1
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("1. Background Knowledge: What Is Our Project Doing?")
    r1.font.color.rgb = RGBColor(14, 116, 144)

    p = doc.add_paragraph()
    p.add_run("• What is a Research Paper? ").bold = True
    p.add_run("When scientists discover something new (like a new vaccine or a new AI model), they write a research paper to share it with the world.\n")
    p.add_run("• What is a Citation? ").bold = True
    p.add_run("When other scientists read your paper and use your discovery, they cite you in their bibliography. Citations are academic 'likes' or 'retweets'. The more citations a paper gets, the more prestigious it becomes.\n")
    p.add_run("• What is Scientometrics? ").bold = True
    p.add_run("The science of measuring academic and scientific impact using math, networks, and statistics.\n")
    p.add_run("• Our Project's Mission: ").bold = True
    p.add_run("When a new paper comes out, nobody knows if it will become world-famous. We freeze all data at exactly 2 years after publication (T_cutoff = Year + 2). Using only what is known at that 2-year mark, our AI predicts whether that paper will achieve Low, Medium, or High citation impact over a 5-year horizon!\n\n")

    p.add_run("The 'Plastic Toy Cake in the Bakery' Analogy:\n").bold = True
    p.add_run("Imagine visiting a luxury bakery. The sign claims an AI Master Chef analyzes gourmet ingredients and critic reviews to bake 5-star cakes. But behind the door in the old codebase:\n"
              "• There was NO chef and NO real graph. The graph function returned raw inputs unchanged (a plain flat MLP)!\n"
              "• It guessed journal prestige by searching for words like 'nature' or 'cell' in the title.\n"
              "• It invented author reputations with a fake math equation: sqrt(citations) * 1.1.\n"
              "• It generated topic velocities with a salted Python hash function that gave a different random number every time the server restarted.\n"
              "• If the teacher asked: 'Where does the model use Open Access?', the honest answer was: 'It doesn't.'\n\n"
              "What We Did: We threw away the plastic toy and built a real scientific laboratory. We pulled 16,500 real OpenAlex papers across 5 fields, connected 649,578 SCImago journal rows by ISSN-L barcode, connected 11,523 CORE conference rankings, and built an authentic 2-hop PyTorch Geometric Graph Neural Network.")

    # SECTION 2: 10 MAJOR UPGRADES TABLE
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. The 10 Major Upgrades (Claims vs. Reality)")
    r2.font.color.rgb = RGBColor(14, 116, 144)

    table_data = [
        ("Feature / Area", "Old Code (The Fake Shortcut)", "New Remediated Code (The Real Science)"),
        ("1. Model Inputs", "5 plain numbers: year, citations (twice), title length, author count. No graph or venue data.", "45 cutoff features: 2y citations, venue SJR, CORE rank, 5 OA one-hots, DOAJ APC, text SVD."),
        ("2. Graph SAGE", "aggregate_neighbors was never given edges. Returned h unchanged (plain MLP).", "Authentic PyG HeteroSAGE with 2-hop message passing across 5 node types & 10 edge types."),
        ("3. Venue Quartiles", "Keyword matching: 'cell' matched 'Cellular', 'acl' matched 'Oracle'. Unknown defaulted to Q2.", "Joined 649k SCImago rows by ISSN-L and 11.5k CORE conference ranks. Unknowns return null."),
        ("4. Author Reputation", "Invented formula: sqrt(citations) * 1.1. Completely fake.", "Real publication histories from OpenAlex across 104,906 real author profiles."),
        ("5. Topic Velocity", "hash(topic) % 120. Salted hash that changed randomly on every server reboot.", "Real field and subfield embeddings computed from actual text and topic ontology."),
        ("6. Open Access Impact", "Hardcoded fake constants (+28.4%, +31.2%) with -$35 paywall penalty.", "Empirical analysis across 16,500 works (+19.3% median boost, 95% CI [+55, +75] citations)."),
        ("7. Dataset Size", "Only 50 papers. Server returned fake fallbacks (50, 257, 215, 2170) if files missing.", "16,500 authentic multi-disciplinary papers, 1.76M citation edges, 7,151 institutions."),
        ("8. Time Leakage", "Used future citations to predict future citations (cheating / lookahead).", "Strict T_cutoff = Year + 2. Zero future data enters. Clean temporal split (train <=2016, test >=2018)."),
        ("9. Error Handling", "Silent 'except Exception: pass' returned empty {} and turned errors into fake 404s.", "Loud failure at boot if model is missing; discrete HTTP codes (404, 429, 502, 503, 422)."),
        ("10. Feature Importance", "Arbitrary handwritten formulas (e.g. w_cit = 0.3 + 0.3*...).", "Real tree-gain feature importance from LightGBM and gradient sensitivity from PyG.")
    ]

    t = doc.add_table(rows=len(table_data), cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_idx, row in enumerate(t.rows):
        for col_idx, cell in enumerate(row.cells):
            cell.text = table_data[row_idx][col_idx]
            if row_idx == 0:
                set_cell_background(cell, "0E7490")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if row_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")
                cell.paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph()

    # SECTION 3: IN-DEPTH TEACHER'S 4 SUGGESTIONS
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. In-Depth Breakdown: The Teacher's 4 Big Suggestions")
    r3.font.color.rgb = RGBColor(14, 116, 144)

    p3_1 = doc.add_paragraph()
    p3_1.add_run("Suggestion 1: Journal Quartiles & CORE Conferences\n").bold = True
    p3_1.add_run("• What was wrong: ").bold = True
    p3_1.add_run("The code matched random words. 'cell' matched 'Cellular Pathology', 'acl' matched 'Oracle'. It defaulted unknown venues to Q2 and missing data to is_q1=True. Even worse, Computer Science conferences (CVPR, NeurIPS) were called 'Q1 journals'.\n")
    p3_1.add_run("• What we did: ").bold = True
    p3_1.add_run("1. Downloaded the official SCImago catalog (649,578 longitudinal rows). Every authentic journal has an 8-digit barcode called an ISSN-L. In ml/scientometrics/venue_resolver.py, we look up that exact barcode for the paper's publication year to get its true quartile (Q1, Q2, Q3, Q4) and SJR score.\n"
                  "2. In Computer Science, conferences are ranked using official CORE rankings (A*, A, B, C). We ingested 11,523 official CORE conference ranks and match conference acronyms (e.g. CVPR, NeurIPS) directly.\n"
                  "3. Strict Null Handling (Rule R1): If a venue is unknown or missing, our system strictly returns null. We never guess Q2 or assume Q1!")

    p3_2 = doc.add_paragraph()
    p3_2.add_run("Suggestion 2: Paid vs. Unpaid Access Economics (APC)\n").bold = True
    p3_2.add_run("• What was wrong: ").bold = True
    p3_2.add_run("The old codebase hardcoded: reader_cost: '$35.00+ / paper' for closed papers and made up author costs.\n")
    p3_2.add_run("• What we did: ").bold = True
    p3_2.add_run("1. Built ml/scientometrics/access_economics.py using real OpenAlex metadata: is_oa (boolean) and oa_status (gold, green, hybrid, bronze, closed).\n"
                  "2. Ingested real Article Processing Charges (APC) in USD from the official DOAJ database: apc_list (catalog price) and apc_paid (actual fee paid by authors).\n"
                  "3. In Gold Open Access, the author pays an APC upfront (e.g., $2,500) so that students worldwide can read it for free. In Closed Access, the author pays $0 upfront, but readers must subscribe. Our backend returns the exact USD fee paid (or null), with zero fake '$35' constants.")

    p3_3 = doc.add_paragraph()
    p3_3.add_run("Suggestion 3: The Open Access Citation Advantage (+19.3% Empirical Analysis)\n").bold = True
    p3_3.add_run("• What was wrong: ").bold = True
    p3_3.add_run("The old code had hardcoded text saying '+28.4% boost' with zero research backing. In real science, this is heavily debated because elite scientists might simply choose Open Access more often (self-selection bias).\n")
    p3_3.add_run("• What we did: ").bold = True
    p3_3.add_run("1. Created scripts/compute_oa_advantage.py and ran an empirical statistical analysis across all 16,500 papers in our dataset, comparing Open Access vs. Closed papers within the same publication year and field.\n"
                  "2. In our dataset, Open Access papers demonstrated a +19.3% median citation advantage (95% bootstrap confidence interval: [+55.0, +75.0] citations).\n"
                  "3. Scientific Rigor: In our API and UI, we explicitly label this: 'Association, not causation (subject to author self-selection bias)'. Teachers love this because it proves scientific honesty!")

    p3_4 = doc.add_paragraph()
    p3_4.add_run("Suggestion 4: Bidirectional Citation Lineage (Inflow vs. Outflow)\n").bold = True
    p3_4.add_run("• What was wrong: ").bold = True
    p3_4.add_run("The old code invented a formula called Knowledge Export Ratio = total citations / reference count, which was mathematically meaningless and cheated by using future citations. It also showed only a chain of the first 5 authors.\n")
    p3_4.add_run("• What we did: ").bold = True
    p3_4.add_run("1. Built ml/scientometrics/lineage_engine.py to model genuine directional citation networks:\n"
                  "   • Outflow (Precursor References / Ancestors): The foundational works cited in the paper's bibliography. We fetch their real titles, years, and citations.\n"
                  "   • Inflow (Adopter Citations / Children): Subsequent papers that cite this target work.\n"
                  "2. Replaced the fake ratio with OpenAlex's official citation_normalized_percentile and the scientometric CD Disruption Index (measuring whether a paper disrupted or consolidated its field).")

    # SECTION 4: ABBREVIATIONS & TECHNOLOGIES
    h4 = doc.add_heading(level=1)
    r4 = h4.add_run("4. Technology & Abbreviation Dictionary (Everyday Analogies)")
    r4.font.color.rgb = RGBColor(14, 116, 144)

    tech_items = [
        ("OpenAlex", "The Wikipedia/Google of all scientific literature on Earth (250M+ papers). We ingested 16,500 real papers from it."),
        ("DOI (Digital Object Identifier)", "The Aadhaar card / passport number of a research paper (e.g. 10.1038/nature12373). Uniquely identifies any paper on Earth."),
        ("ISSN / ISSN-L", "The barcode printed on every journal. We match this barcode against 649,000 SCImago records to find true journal prestige."),
        ("SCImago & SJR", "The ICC / FIFA rankings for journals. Divides journals into Q1 (top 25% premier elite), Q2 (top 25-50%), Q3, and Q4."),
        ("CORE Conference Rankings", "The official report-card ranking for Computer Science conferences (A*, A, B, C). A* is the top 5% premier venues like CVPR and NeurIPS."),
        ("Open Access (OA) & APC", "Open Access means free to read online. APC (Article Processing Charge) is the fee the authors paid to make it free (from DOAJ)."),
        ("Parquet (.parquet)", "High-speed, compressed file format created by big tech. Stores millions of rows and loads in milliseconds without crashing."),
        ("MLP (Multi-Layer Perceptron)", "A standard flat neural network. It takes numbers in a straight row with no understanding of networks or connections."),
        ("GNN (Graph Neural Network)", "An AI designed to learn from webs of connected dots (nodes) and lines (edges), like social networks or citation webs."),
        ("PyG (PyTorch Geometric)", "The world's leading Python framework for building and training Graph Neural Networks on graph data."),
        ("Heterogeneous Graph", "A graph with MULTIPLE types of dots (papers, authors, universities, journals, topics) and relations between them."),
        ("GraphSAGE & 2-Hop", "An algorithm where a paper asks its direct friends for advice (Hop 1), and those friends ask their friends (Hop 2)."),
        ("LightGBM", "Microsoft's world-champion decision-tree algorithm. Used as a strong tabular baseline and to calculate real feature attributions."),
        ("Macro-F1 Score", "An unbiased test score evaluating accuracy equally across all three classes (Low, Medium, and High impact)."),
        ("AUROC (Area Under ROC)", "A score from 0.5 (random coin toss) to 1.0 (psychic). Measures how well the AI ranks papers by impact. Our GNN scored 0.7205."),
        ("ECE (Expected Calibration Error)", "Checks if the model's confidence matches reality. Temperature scaling reduced our ECE to an exceptional 0.0083."),
        ("FastAPI & Uvicorn", "The modern high-speed Python web server running our project backend on port 8080."),
        ("Cytoscape.js", "The JavaScript library that draws the interactive, drag-and-drop network graph on the right side of our web screen.")
    ]

    for term, expl in tech_items:
        p = doc.add_paragraph()
        p.add_run(f"• {term}: ").bold = True
        p.add_run(expl)

    # SECTION 5: BENCHMARK RESULTS
    h5 = doc.add_heading(level=1)
    r5 = h5.add_run("5. Scientific Benchmark Results (Held-out Test Set: 4,500 Papers)")
    r5.font.color.rgb = RGBColor(14, 116, 144)

    bench_data = [
        ("Model Paradigm", "Macro-F1", "AUROC", "MAE (Log Cits)", "ECE Calibration"),
        ("Log-Linear Early Citations", "0.3767", "0.6489", "0.4607", "0.3584"),
        ("Logistic Regression (L2)", "0.5057", "0.7180", "0.4365", "0.1214"),
        ("LightGBM Boosted Trees", "0.4995", "0.7179", "0.4185 (Best)", "0.1832"),
        ("Graphless Deep MLP", "0.5026", "0.7153", "0.6633", "0.0618"),
        ("★ PyG HeteroGraphSAGE (2-Hop)", "0.5381 (Best)", "0.7205 (Best)", "0.5088", "0.0083 (Best)")
    ]

    tb = doc.add_table(rows=len(bench_data), cols=5)
    tb.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row_idx, row in enumerate(tb.rows):
        for col_idx, cell in enumerate(row.cells):
            cell.text = bench_data[row_idx][col_idx]
            if row_idx == 0:
                set_cell_background(cell, "0E7490")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)
            elif row_idx == 5:
                set_cell_background(cell, "E0F2FE")
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.bold = True
            else:
                if row_idx % 2 == 1:
                    set_cell_background(cell, "F8FAFC")
                cell.paragraphs[0].runs[0].font.size = Pt(9)

    doc.add_paragraph()

    # SECTION 6: SCRIPT
    h6 = doc.add_heading(level=1)
    r6 = h6.add_run("6. Your 3-Minute Presentation Script (Say This Word-for-Word)")
    r6.font.color.rgb = RGBColor(14, 116, 144)

    p_sc = doc.add_paragraph()
    p_sc.paragraph_format.left_indent = Inches(0.3)
    p_sc.paragraph_format.right_indent = Inches(0.3)
    p_sc.add_run(
        '"Good morning, respected teachers. Today we present SciGraph AI 2.0: Predicting 5-Year Citation Trajectories Using Heterogeneous Graph Neural Networks.\n\n'
        'Our project addresses a core scientometric challenge: Can an AI accurately predict whether a newly published research paper will achieve Low, Medium, or High citation impact over a 5-year horizon, using strictly what is known 2 years after its publication?\n\n'
        'We remediated this project under 4 strict scientific principles:\n'
        '1. Zero Time Leakage: Every citation and indicator is locked at T_cutoff = Year + 2. We never peek into future citations.\n'
        '2. Zero Fabricated Data: 649,000 SCImago journal records by ISSN-L and 11,500 CORE conference rankings. Unindexed venues return null.\n'
        '3. Authentic Access Economics: Real DOAJ APC fees and an empirical +19.3% median OA citation advantage across 16,500 works.\n'
        '4. Authentic Heterogeneous GNN: 2-hop message passing across 5 node types in PyTorch Geometric, achieving 0.5381 Macro-F1 on held-out temporal test papers, outperforming graphless MLPs and linear baselines.\n\n'
        'Now, let us demonstrate the live dashboard and backtest verification."'
    ).italic = True

    doc.save(filename)
    print(f"Generated Updated DOCX: {filename}")


if __name__ == "__main__":
    create_docx()
