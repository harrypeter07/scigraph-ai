# Known Issues & Technical Debt Registry

> **Purpose**: Active tracking of technical debt, unresolved bugs, API constraints, and deferred work items.

---

## Active Tracking List

| Issue ID | Category | Description | Workaround / Mitigation | Status |
|---|---|---|---|---|
| **ISSUE-001** | API Rate Limiting | OpenAlex public endpoint may return HTTP 429 if request volume bursts > 10 req/sec. | Enforced `0.1s` sleep in acquisition loop with exponential backoff. | Open / Mitigated |
| **ISSUE-002** | Compute Constraint | Laptop CPU cannot perform full GNN training on large graph topologies. | GNN training moved to Lab GPU PCs (Phase 8); CPU limited to smoke test on synthetic graphs. | Active Design Constraint |
