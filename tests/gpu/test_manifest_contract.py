import copy
import math

import pytest
from app.gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest

OBJECT_INFO = {
    "DemoNode": {
        "input": {
            "required": {
                "text": ["STRING"],
                "steps": ["INT", {"default": 20, "min": 1, "max": 50}],
                "seed": ["INT", {"default": 1, "min": 0, "max": 2**31 - 1}],
            },
            "optional": {"image": ["IMAGE"]},
        },
        "output": ["IMAGE"],
    }
}
API_GRAPH = {"1": {"class_type": "DemoNode", "inputs": {"text": "hello", "steps": 20, "seed": 1}}}
MANIFEST = {
    "model_id": "demo",
    "parameters": {
        "prompt": {"type": "string", "default": "hello", "binding": {"node": "1", "input": "text"}},
        "steps": {
            "type": "integer",
            "default": 20,
            "min": 1,
            "max": 50,
            "binding": {"node": "1", "input": "steps"},
        },
        "seed": {
            "type": "integer",
            "default": 1,
            "min": -1,
            "max": 2**31 - 1,
            "binding": {"node": "1", "input": "seed"},
        },
    },
}


def test_validate_manifest_binds_only_real_object_info_inputs():
    result = validate_manifest(MANIFEST, API_GRAPH, OBJECT_INFO)
    assert result["parameters"]["prompt"]["binding"] == {"node": "1", "input": "text"}
    assert result["parameters"]["steps"]["type"] == "integer"


def test_validate_manifest_rejects_unknown_node_and_connected_input():
    with pytest.raises(ValueError, match="未知|unknown"):
        validate_manifest(
            {
                **MANIFEST,
                "parameters": {
                    **MANIFEST["parameters"],
                    "x": {"type": "string", "binding": {"node": "404", "input": "text"}},
                },
            },
            API_GRAPH,
            OBJECT_INFO,
        )
    graph = copy.deepcopy(API_GRAPH)
    graph["1"]["inputs"]["image"] = ["2", 0]
    with pytest.raises(ValueError, match="connected|连接"):
        validate_manifest(
            {
                **MANIFEST,
                "parameters": {
                    **MANIFEST["parameters"],
                    "image": {"type": "image", "binding": {"node": "1", "input": "image"}},
                },
            },
            graph,
            OBJECT_INFO,
        )


def test_resolve_parameters_applies_preset_and_rejects_unknown_or_invalid_values():
    resolved = resolve_parameters(MANIFEST, {"prompt": "world", "seed": -1}, {"steps": 30})
    assert resolved["prompt"] == "world" and resolved["steps"] == 30
    assert isinstance(resolved["seed"], int) and 0 <= resolved["seed"] <= 2**31 - 1
    with pytest.raises(ValueError, match="未知|unknown"):
        resolve_parameters(MANIFEST, {"nope": 1})
    with pytest.raises(ValueError, match="范围|range|类型|type"):
        resolve_parameters(MANIFEST, {"steps": 99})
    with pytest.raises(ValueError, match="类型|type"):
        resolve_parameters(MANIFEST, {"steps": "20"})


def test_bind_parameters_deep_copies_graph_and_writes_resolved_values():
    graph = copy.deepcopy(API_GRAPH)
    bound = bind_parameters(MANIFEST, graph, {"prompt": "bound", "steps": 33, "seed": -1})
    assert bound["1"]["inputs"]["text"] == "bound" and bound["1"]["inputs"]["steps"] == 33
    assert graph == API_GRAPH


def test_digest_is_canonical():
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    assert len(digest({})) == 64


def test_validate_manifest_accepts_canonical_parameter_and_path_bindings():
    manifest = {
        "version": "1.2.3",
        "parameters": [
            {"name": "prompt", "type": "string", "default": "hello"},
            {"name": "steps", "type": "integer", "default": 20},
        ],
        "bindings": [
            {"parameter": "prompt", "node": "1", "path": "inputs.text"},
            {"parameter": "steps", "node": "1", "path": "inputs.steps"},
        ],
    }
    result = validate_manifest(manifest, API_GRAPH, OBJECT_INFO)
    assert isinstance(result["parameters"], dict)
    assert result["parameters"]["prompt"]["binding"] == {"node": "1", "input": "text"}
    assert resolve_parameters(result, {"prompt": "world", "steps": 21})["steps"] == 21


def test_validate_manifest_checks_every_node_and_link_type():
    graph = {
        "1": {"class_type": "DemoNode", "inputs": {"text": "hello", "steps": 20, "seed": 1}},
        "2": {"class_type": "MissingNode", "inputs": {}},
    }
    with pytest.raises(ValueError, match="节点类型|未知"):
        validate_manifest(MANIFEST, graph, OBJECT_INFO)

    graph = copy.deepcopy(API_GRAPH)
    graph["1"]["inputs"]["image"] = ["404", 0]
    with pytest.raises(ValueError, match="源节点|unknown|未知"):
        validate_manifest(MANIFEST, graph, OBJECT_INFO)

    graph = {
        "1": {"class_type": "DemoNode", "inputs": {"text": "hello", "steps": 20, "seed": 1}},
        "2": {"class_type": "DemoNode", "inputs": {"text": ["1", 0], "steps": 20, "seed": 1}},
    }
    with pytest.raises(ValueError, match="类型|输出"):
        validate_manifest(MANIFEST, graph, OBJECT_INFO)


def test_object_info_bounds_apply_when_manifest_omits_bounds_and_reject_nonfinite():
    manifest = copy.deepcopy(MANIFEST)
    manifest["parameters"]["steps"].pop("min")
    manifest["parameters"]["steps"].pop("max")
    normalized = validate_manifest(manifest, API_GRAPH, OBJECT_INFO)
    with pytest.raises(ValueError, match="范围"):
        resolve_parameters(normalized, {"steps": 0})
    with pytest.raises(ValueError, match="有限|finite|类型"):
        resolve_parameters(normalized, {"prompt": "x", "steps": math.inf, "seed": 1})


def test_seed_minus_one_is_resolved_to_a_real_integer(monkeypatch):
    monkeypatch.setattr("app.gpu.manifest.secrets.randbelow", lambda size: size - 1)
    normalized = validate_manifest(MANIFEST, API_GRAPH, OBJECT_INFO)
    resolved = resolve_parameters(normalized, {"prompt": "x", "steps": 20, "seed": -1})
    assert resolved["seed"] == 2**31 - 1
    assert isinstance(resolved["seed"], int)


def test_resolve_parameters_rejects_empty_required_image():
    manifest = {
        "model_id": "demo",
        "parameters": {
            "ref_1": {"type": "image", "label": "参考图 1", "required": True, "binding": {"node": "1", "input": "image"}},
            "ref_2": {"type": "image", "label": "参考图 2", "required": False, "binding": {"node": "1", "input": "image"}},
        },
    }
    graph = {"1": {"class_type": "DemoNode", "inputs": {"image": "default.jpg"}}}
    normalized = validate_manifest(manifest, graph, OBJECT_INFO)

    # Empty string for required image must fail immediately
    with pytest.raises(ValueError, match="缺少必填"):
        resolve_parameters(normalized, {"ref_1": "", "ref_2": ""})

    # Valid image for required, empty for optional should succeed and drop ref_2
    resolved = resolve_parameters(normalized, {"ref_1": "asset_123.jpg", "ref_2": ""})
    assert resolved["ref_1"] == "asset_123.jpg"
    assert "ref_2" not in resolved
