"""A1 normative document RED; candidate approved for test baseline only.

E8 residual-risk acceptance is NOT supplied by these tests. Expected complete
clauses are frozen below; runtime never reads external audit artifacts.
"""
from pathlib import Path
import re

import pytest

CANDIDATE_SHA256 = "bfff8b03fdc722da445f8ad6b551881c06e21b7e7a0f299f3277ee41645aa36e"
CONTRACT_ID = "m16-m17-legacy-trusted-host-python-v1"
HEADING = "## M16/M17 Legacy Trusted-Host Python API (v1)"
REFERENCE = "v1_security.md#" + CONTRACT_ID
CANONICAL_CLAUSES = {'R01': 'M16/M17 Legacy Trusted-Host Python API v1 applies to analyze_experiment_evidence, compare_experiment_evidence, ExperimentEvidenceTools and ExperimentComparisonTools. Host trust does not establish automatic object safety.', 'R02': 'M16/M17 do not guarantee callback-free rejection of arbitrary Python objects. Type checks and semantic validation may invoke supplied object protocols before accepting or rejecting an input.', 'R03': 'Before any Session invocation, M16 Tool construction may call deepcopy and M17 Tool construction may call mapping methods, dict(options) and deepcopy. The host must review these protocols before constructing the Tool.', 'R04': 'Deepcopy is cooperative copying, not an object-isolation security boundary. A custom deepcopy implementation can return the original object or retain aliases; the host must ensure bound selections remain stable and must not assume arbitrary objects were detached.', 'R05': 'Ordinary model tool arguments are decoded as JSON data and cannot directly instantiate arbitrary Python classes or replace host-bound config_paths, test_metric_specs, targets or options. Python SDK response objects and host extensions remain a separate executable trust surface.', 'R06': 'An authorized model request may invoke protocols of objects previously bound by the host. This is host-object reachability, not direct model injection of those Python objects, and it does not establish their safety.', 'R07': 'Session path authorization applies to invocation and precedes the relevant bound Tool dispatch. It does not protect earlier constructor protocol execution, and Session history rollback cannot undo Python callback side effects.', 'R08': 'M18 and M19 have separately scoped exact-type, fixed-field and bounded snapshot contracts. M16/M17 do not inherit those callback-rejection or input-bound guarantees; a bounded output does not establish a uniform pre-traversal input bound.', 'R09': 'Direct Python Core, Reader, Tool and Adapter callers must establish their own applicable authority and input trust. A direct Adapter call does not create a missing Session path policy. Host-selected filesystem authority is separate from Python object-protocol trust.', 'R10': 'Unreviewed plugin, proxy, lazy-provider, custom SDK or other dynamic objects are outside the reviewed safe input domain. External Python extension interfaces remain usable under host responsibility, but this V1 acceptance does not certify external integrations or arbitrary-object safety.'}


def _normalized(text):
    return " ".join(text.split())


def _clause(document, clause_id):
    opening = "<!-- LTH-V1:" + clause_id + " -->"
    closing = "<!-- /LTH-V1:" + clause_id + " -->"
    assert document.count(opening) == 1, f"{clause_id}: missing or duplicate opening contract marker"
    assert document.count(closing) == 1, f"{clause_id}: missing or duplicate closing contract marker"
    start = document.index(opening) + len(opening)
    end = document.index(closing)
    assert end >= start, f"{clause_id}: reversed contract markers"
    return document[start:end]


def assert_contract(documents, clause_id):
    """Exact normative text plus named scope/version and real Markdown links."""
    security = documents["v1_security.md"]
    actual = _clause(security, clause_id)
    assert _normalized(actual) == _normalized(CANONICAL_CLAUSES[clause_id]), (
        f"{clause_id}: normative contract text mismatch")
    assert security.splitlines().count(HEADING) == 1, f"{clause_id}: missing/wrong API scope or version"
    anchor = '<a id="' + CONTRACT_ID + '"></a>'
    assert security.count(anchor) == 1, f"{clause_id}: missing/duplicate versioned anchor"
    section_start = security.index(HEADING)
    next_section = re.search(r"^## ", security[section_start + len(HEADING):], re.MULTILINE)
    section_end = (section_start + len(HEADING) + next_section.start()
                   if next_section else len(security))
    assert section_start < security.index("<!-- LTH-V1:" + clause_id + " -->") < section_end, (
        f"{clause_id}: clause outside named legacy contract section")
    for name in ("v1_api_reference.md", "v1_architecture.md"):
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", documents[name])
        assert REFERENCE in links, f"{clause_id}: missing versioned contract reference in {name}"


def calibrate_assertions():
    """Run independently from pytest RED; all mutations are memory-only."""
    security = '<a id="' + CONTRACT_ID + '"></a>\n' + HEADING + "\n\n"
    for key, text in CANONICAL_CLAUSES.items():
        security += f"<!-- LTH-V1:{key} -->\n{text}\n<!-- /LTH-V1:{key} -->\n\n"
    documents = {"v1_security.md": security,
                 "v1_api_reference.md": f"[Legacy contract]({REFERENCE})",
                 "v1_architecture.md": f"[Legacy contract]({REFERENCE})"}
    records = []
    for key, text in CANONICAL_CLAUSES.items():
        assert_contract(documents, key)
        harmless = dict(documents)
        harmless["v1_security.md"] = security.replace(text, " \n  ".join(text.split()))
        assert_contract(harmless, key)
        mutations = {
            "missing": security.replace(f"<!-- LTH-V1:{key} -->", ""),
            "reversed-guarantee": security.replace(text, "The opposite of this guarantee applies: " + text),
            "wrong-version": security.replace(HEADING, HEADING.replace("(v1)", "(v2)")),
            "wrong-scope": security.replace(HEADING, HEADING.replace("M16/M17", "M18/M19")),
            "duplicate-marker": security + f"<!-- LTH-V1:{key} -->",
        }
        for kind, mutated in mutations.items():
            sample = dict(documents, **{"v1_security.md": mutated})
            with pytest.raises(AssertionError, match=key):
                assert_contract(sample, key)
            records.append({"clause": key, "mutation": kind, "rejected": True})
        records.append({"clause": key, "mutation": "whitespace-only", "accepted": True})
    return records


@pytest.mark.parametrize("clause_id", list(CANONICAL_CLAUSES))
def test_versioned_legacy_contract(clause_id):
    root = Path(__file__).resolve().parents[1] / "docs"
    documents = {name: (root / name).read_text(encoding="utf-8")
                 for name in ("v1_security.md", "v1_api_reference.md", "v1_architecture.md")}
    assert_contract(documents, clause_id)
