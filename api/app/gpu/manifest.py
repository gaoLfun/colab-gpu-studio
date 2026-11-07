"""Schema-first manifest validation and safe graph parameter binding."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import secrets
from collections.abc import Mapping
from typing import Any

_DANGEROUS = {
    "python",
    "exec",
    "eval",
    "shell",
    "subprocess",
    "command",
    "bash",
    "powershell",
    "httprequest",
    "download",
    "websocket",
}
_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def digest(value: dict) -> str:
    try:
        _finite(value)
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest只能包含JSON兼容值和有限数值") from exc
    return hashlib.sha256(raw).hexdigest()


def _fail(message: str):
    raise ValueError(message)


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail("数值必须是有限值")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("对象键必须是字符串")
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _type(value: Any) -> str:
    aliases = {
        "str": "string",
        "text": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "image": "image",
        "video": "video",
        "asset": "asset",
        "enum": "enum",
        "choice": "enum",
        "array": "array",
        "list": "array",
        "object": "object",
        "json": "json",
        "color": "string",
        "any": "any",
    }
    if not isinstance(value, str) or value.lower() not in aliases:
        _fail(f"不支持的参数类型: {value!r}")
    return aliases[value.lower()]


def _node(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail("节点ID格式无效")
    return str(value)


def _binding(name: str, value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        node, _, path = value.partition(".")
        value = {"node": node, "path": path or value}
    if isinstance(value, Mapping):
        node = value.get("node", value.get("node_id"))
        inp = value.get("input", value.get("input_name"))
        path = value.get("path")
        if isinstance(path, str):
            parts = path.split(".")
            if len(parts) == 3 and parts[1] == "inputs":
                if node is not None and str(node) != parts[0]:
                    _fail(f"参数 {name} binding node与path冲突")
                node, inp = parts[0], parts[2]
            elif len(parts) == 2 and parts[0] == "inputs":
                inp = parts[1]
            else:
                _fail(f"参数 {name} binding path格式无效")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        node, inp = value
    else:
        _fail(f"参数 {name} binding格式无效")
    if node is None or not isinstance(inp, str) or not inp or "." in inp:
        _fail(f"参数 {name} binding格式无效")
    return _node(node), inp


def _canonical_parameters(manifest: Mapping) -> tuple[dict, dict]:
    """Return parameter definitions as a dict and the manifest copy.

    The public/canonical form uses ``parameters`` as a list and puts paths in a
    separate ``bindings`` list.  The executor deliberately consumes only the
    normalized dictionary form, while accepting the old inline dictionary form.
    """
    normalized = copy.deepcopy(dict(manifest))
    raw = normalized.get("parameters")
    if isinstance(raw, Mapping):
        definitions = copy.deepcopy(dict(raw))
    elif isinstance(raw, list):
        definitions = {}
        for item in raw:
            if not isinstance(item, Mapping):
                _fail("参数定义格式无效")
            name = item.get("name", item.get("parameter", item.get("id")))
            if not isinstance(name, str) or not name:
                _fail("参数定义缺少name")
            if name in definitions:
                _fail(f"重复参数: {name}")
            definitions[name] = {
                k: copy.deepcopy(v) for k, v in item.items() if k not in {"name", "parameter", "id"}
            }
    else:
        _fail("manifest.parameters不能为空")
    if not definitions:
        _fail("manifest.parameters不能为空")

    raw_bindings = normalized.get("bindings", [])
    if isinstance(raw_bindings, Mapping):
        binding_items = [
            {"parameter": name, **(value if isinstance(value, Mapping) else {})}
            for name, value in raw_bindings.items()
        ]
    elif isinstance(raw_bindings, list):
        binding_items = raw_bindings
    else:
        _fail("manifest.bindings格式无效")
    for item in binding_items:
        if not isinstance(item, Mapping):
            _fail("binding定义格式无效")
        name = item.get("parameter", item.get("name", item.get("param")))
        if name not in definitions:
            _fail(f"binding引用未知参数: {name!r}")
        binding = dict(item)
        binding.pop("parameter", None)
        binding.pop("name", None)
        binding.pop("param", None)
        if "binding" in binding:
            binding = binding["binding"]
        old = definitions[name].get("binding")
        if old is not None:
            if _binding(name, old) != _binding(name, binding):
                _fail(f"参数 {name}存在冲突binding")
        else:
            definitions[name]["binding"] = binding
    normalized["parameters"] = definitions
    return definitions, normalized


def _connected(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _object_input(object_info: Mapping, class_type: str, input_name: str):
    info = object_info.get(class_type)
    if not isinstance(info, Mapping):
        _fail(f"未知节点类型: {class_type}")
    inputs = info.get("input", {})
    for section in ("required", "optional", "hidden"):
        section_value = inputs.get(section, {}) if isinstance(inputs, Mapping) else {}
        if isinstance(section_value, Mapping) and input_name in section_value:
            spec = section_value[input_name]
            if isinstance(spec, (list, tuple)) and spec:
                if isinstance(spec[0], (list, tuple)):
                    options = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], Mapping) else {}
                    options.setdefault("values", list(spec[0]))
                    return "ENUM", options
                return str(spec[0]).upper(), dict(spec[1]) if len(spec) > 1 and isinstance(
                    spec[1], Mapping
                ) else {}
            if isinstance(spec, str):
                return spec.upper(), {}
            if isinstance(spec, Mapping):
                return str(spec.get("type", "")).upper(), dict(spec)
    _fail(f"节点 {class_type} 不存在输入 {input_name}")


def _outputs(info: Mapping) -> list[str]:
    raw = info.get("output", info.get("output_types", info.get("outputs", [])))
    if isinstance(raw, Mapping):
        raw = raw.get("types", raw.get("output_types", list(raw.values())))
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        _fail("节点output格式无效")
    return [str(item).upper() for item in raw]


def _compatible(source: str, target: str) -> bool:
    source, target = source.upper(), target.upper()
    if target in {"", "ANY", "JSON", "OBJECT", "DICT"} or source in {"ANY", "*"}:
        return True
    if source == target:
        return True
    return (
        target in {"ASSET", "IMAGE", "VIDEO"} and source in {"IMAGE", "VIDEO", "STRING"} and target == "ASSET"
    )


def _check(name: str, spec: Mapping, value: Any) -> Any:
    _finite(value)
    kind = _type(spec.get("type"))
    if kind == "string" and not isinstance(value, str):
        _fail(f"参数 {name} 类型错误")
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        _fail(f"参数 {name} 类型错误")
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
        _fail(f"参数 {name} 类型错误")
    if kind == "boolean" and not isinstance(value, bool):
        _fail(f"参数 {name} 类型错误")
    if kind == "array" and not isinstance(value, list):
        _fail(f"参数 {name} 类型错误")
    if kind == "object" and not isinstance(value, dict):
        _fail(f"参数 {name} 类型错误")
    if kind in {"image", "video", "asset"}:
        if not isinstance(value, (str, list, tuple, dict)):
            _fail(f"参数 {name} 类型错误")
        if spec.get("required") and (value is None or (isinstance(value, str) and not value.strip())):
            label = spec.get("label") or name
            _fail(f"缺少必填素材: {label}")
    choices = spec.get("choices", spec.get("enum"))
    if choices is not None and value not in choices:
        _fail(f"参数 {name} 不在允许选项中")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        random_seed = name.lower() == "seed" and value == -1 and kind == "integer"
        if not random_seed:
            if spec.get("min") is not None and value < spec["min"]:
                _fail(f"参数 {name} 超出范围")
            if spec.get("max") is not None and value > spec["max"]:
                _fail(f"参数 {name} 超出范围")
    return value


def _graph_key(graph: Mapping, node: str):
    if node in graph:
        return node
    return next((key for key in graph if str(key) == node), None)


def _validate_graph(graph: Mapping, object_info: Mapping) -> set[str]:
    node_ids = {str(key) for key in graph}
    if not node_ids:
        _fail("api_graph不能为空")
    for key, node in graph.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("inputs"), Mapping):
            _fail(f"节点 {key}格式无效")
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or class_type not in object_info:
            _fail(f"节点类型未知: {class_type!r}")
        compact = class_type.lower().replace("_", "")
        if any(token in compact for token in _DANGEROUS):
            _fail(f"危险节点类型禁止使用: {class_type}")
        info = object_info[class_type]
        if not isinstance(info, Mapping):
            _fail(f"节点类型object_info格式无效: {class_type}")
        for input_name, value in node["inputs"].items():
            if not isinstance(input_name, str):
                _fail("节点输入名格式无效")
            actual, _ = _object_input(object_info, class_type, input_name)
            link_shaped = (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and isinstance(value[0], (str, int))
                and not isinstance(value[0], bool)
            )
            if link_shaped and actual not in {"LIST", "ARRAY"} and not _connected(value):
                _fail(f"输入 {key}.{input_name} 的connected link格式无效")
            if _connected(value) and actual not in {"LIST", "ARRAY"}:
                source_key = _graph_key(graph, _node(value[0]))
                if source_key is None:
                    _fail(f"输入 {key}.{input_name} 的connected源节点不存在")
                source = graph[source_key]
                source_info = object_info[source["class_type"]]
                outputs = _outputs(source_info)
                slot = value[1]
                if slot < 0 or slot >= len(outputs):
                    _fail(f"输入 {key}.{input_name} 的output_slot不存在")
                if not _compatible(outputs[slot], actual):
                    _fail(f"输入 {key}.{input_name} 的连接类型不匹配")
    return node_ids


def validate_manifest(manifest: dict, api_graph: dict, object_info: dict) -> dict:
    if not all(isinstance(value, Mapping) for value in (manifest, api_graph, object_info)):
        _fail("manifest、api_graph、object_info必须是对象")
    _finite(manifest)
    definitions, normalized = _canonical_parameters(manifest)
    if "version" in normalized and (
        not isinstance(normalized["version"], str) or not _SEMVER.fullmatch(normalized["version"])
    ):
        _fail("manifest.version必须是semver")
    _validate_graph(api_graph, object_info)
    graph = copy.deepcopy(dict(api_graph))
    output = {}
    compatible = {
        "string": {"STRING", "STR"},
        "integer": {"INT", "INTEGER"},
        "number": {"FLOAT", "NUMBER", "INT"},
        "boolean": {"BOOL", "BOOLEAN"},
        "image": {"IMAGE", "ENUM"},  # LoadImage.image is a COMBO of file names
        "video": {"VIDEO"},
        "asset": {"IMAGE", "VIDEO", "STRING"},
        "enum": {"STRING", "ENUM", "COMBO"},
        "array": {"LIST", "ARRAY"},
        "object": {"DICT", "OBJECT", "JSON"},
        "json": {"JSON", "STRING", "DICT", "OBJECT"},
        "any": set(),
    }
    for name, raw in definitions.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            _fail("参数定义格式无效")
        spec = copy.deepcopy(dict(raw))
        kind = _type(spec.get("type"))
        spec["type"] = kind
        if "variants" in spec:
            _check_variants(name, spec, graph, object_info)
            output[name] = spec
            continue
        if "binding" not in spec:
            _fail(f"参数 {name} 必须有binding")
        node, inp = _binding(name, spec["binding"])
        key = _graph_key(graph, node)
        if key is None:
            _fail(f"参数 {name} 绑定未知节点 {node}")
        node_data = graph[key]
        class_type = node_data.get("class_type")
        if inp not in node_data["inputs"]:
            _fail(f"节点 {node} 不存在图输入 {inp}")
        if _connected(node_data["inputs"][inp]):
            _fail(f"参数 {name} 不能绑定connected input")
        actual, options = _object_input(object_info, class_type, inp)
        if actual and actual not in compatible[kind] and kind != "any":
            _fail(f"参数 {name} 类型与真实节点输入不匹配")
        if kind == "enum" and options.get("values") and "choices" not in spec:
            spec["choices"] = list(options["values"])
        object_options = options.get("options") if isinstance(options.get("options"), Mapping) else options
        object_min, object_max = object_options.get("min"), object_options.get("max")
        if object_min is not None:
            if spec.get("min") is None:
                spec["min"] = object_min
            elif spec["min"] < object_min:
                if name.lower() == "seed" and spec["min"] == -1:
                    spec["min"] = object_min
                else:
                    _fail(f"参数 {name} 小于节点允许范围")
        if object_max is not None:
            if spec.get("max") is None:
                spec["max"] = object_max
            elif spec["max"] > object_max:
                _fail(f"参数 {name} 大于节点允许范围")
        if "default" in spec:
            _check(name, spec, spec["default"])
        if spec.get("min") is not None and spec.get("max") is not None and spec["min"] > spec["max"]:
            _fail(f"参数 {name} 范围无效")
        spec["binding"] = {"node": node, "input": inp}
        output[name] = spec
    normalized["parameters"] = output
    normalized["manifest_version"] = int(normalized.get("manifest_version", 1))
    return normalized


def _check_variants(name: str, spec: dict, graph: dict, object_info: Mapping) -> None:
    """A choice that rewrites the graph: each variant sets node inputs (values or links) and drops nodes."""
    variants = spec["variants"]
    if spec["type"] != "enum" or not isinstance(variants, Mapping) or not variants:
        _fail(f"参数 {name} variants格式无效")
    spec["choices"] = list(variants)
    if spec.get("default") not in variants:
        _fail(f"参数 {name} 默认值必须是一个variant")
    for choice in variants:
        applied = copy.deepcopy(graph)
        try:
            _apply_variant(applied, variants[choice])
        except (KeyError, TypeError, AttributeError):
            _fail(f"参数 {name} 的 {choice} 指向不存在的节点或输入")
        _validate_graph(applied, object_info)


def _apply_variant(graph: dict, variant: Mapping) -> None:
    for node, inputs in (variant.get("set") or {}).items():
        target = graph[_graph_key(graph, node)]["inputs"]
        for inp, value in inputs.items():
            if inp not in target:
                raise KeyError(inp)
            target[inp] = copy.deepcopy(value)
    for node in variant.get("drop") or []:
        _drop_node(graph, str(node))


def _normalized_definitions(manifest: Mapping) -> dict:
    definitions, _ = _canonical_parameters(manifest)
    return definitions


def _optional_image(spec: Mapping) -> bool:
    return spec.get("type") == "image" and spec.get("required") is False


def resolve_parameters(manifest: dict, params: dict, preset: dict | None = None) -> dict:
    definitions = _normalized_definitions(manifest) if isinstance(manifest, Mapping) else None
    if not isinstance(definitions, dict) or not isinstance(params, Mapping):
        _fail("manifest.parameters和params必须是对象")
    values_from_preset = preset or {}
    if isinstance(values_from_preset, Mapping) and isinstance(values_from_preset.get("params"), Mapping):
        values_from_preset = values_from_preset["params"]
    if not isinstance(values_from_preset, Mapping):
        _fail("preset参数格式无效")
    unknown = (set(params) | set(values_from_preset)) - set(definitions)
    if unknown:
        _fail(f"未知参数: {sorted(unknown)!r}")
    resolved = {}
    for name, spec in definitions.items():
        if not isinstance(spec, Mapping):
            _fail(f"参数 {name}定义无效")
        if "default" in spec:
            resolved[name] = copy.deepcopy(spec["default"])
        if name in values_from_preset:
            resolved[name] = copy.deepcopy(values_from_preset[name])
        if name in params:
            resolved[name] = copy.deepcopy(params[name])
        if _optional_image(spec) and resolved.get(name) in (None, ""):
            resolved.pop(name, None)  # not given: bind_parameters drops its LoadImage node
            continue
        if name not in resolved or (spec.get("required") and (resolved.get(name) is None or (isinstance(resolved[name], str) and not resolved[name].strip()))):
            label = spec.get("label") or name
            _fail(f"缺少必填参数: {label}")
        value = resolved[name]
        _check(name, spec, value)
        if name.lower() == "seed" and value == -1:
            low = spec.get("min", 0)
            high = spec.get("max")
            if low == -1:
                low = 0
            if high is None or not isinstance(low, int) or not isinstance(high, int) or high < low:
                _fail("seed随机范围无效")
            resolved[name] = secrets.randbelow(high - low + 1) + low
    return resolved


def bind_parameters(manifest: dict, api_graph: dict, resolved: dict) -> dict:
    definitions = _normalized_definitions(manifest) if isinstance(manifest, Mapping) else None
    if (
        not isinstance(definitions, dict)
        or not isinstance(api_graph, Mapping)
        or not isinstance(resolved, Mapping)
    ):
        _fail("manifest、api_graph、resolved格式无效")
    unknown = set(resolved) - set(definitions)
    if unknown:
        _fail(f"未知参数: {sorted(unknown)!r}")
    graph = copy.deepcopy(dict(api_graph))
    for name, spec in definitions.items():
        if name not in resolved or "variants" in spec:
            continue
        if not isinstance(spec, Mapping):
            _fail(f"参数 {name}定义无效")
        node, inp = _binding(name, spec.get("binding"))
        key = _graph_key(graph, node)
        if (
            key is None
            or not isinstance(graph[key], Mapping)
            or not isinstance(graph[key].get("inputs"), Mapping)
        ):
            _fail(f"参数 {name}绑定目标不存在")
        if inp not in graph[key]["inputs"]:
            _fail(f"参数 {name}绑定输入不存在")
        if _connected(graph[key]["inputs"].get(inp)):
            _fail(f"参数 {name}不能绑定connected input")
        _check(name, spec, resolved[name])
        value = resolved[name]
        if name.lower() == "seed" and value == -1:
            high = spec.get("max")
            low = spec.get("min", 0)
            if low == -1:
                low = 0
            if not isinstance(low, int) or not isinstance(high, int) or high < low:
                _fail("seed随机范围无效")
            value = secrets.randbelow(high - low + 1) + low
        graph[key]["inputs"][inp] = copy.deepcopy(value)
    for name, spec in definitions.items():
        if _optional_image(spec) and name not in resolved:
            _drop_node(graph, _binding(name, spec.get("binding"))[0])
    for name, spec in definitions.items():
        if "variants" in spec and name in resolved:
            _apply_variant(graph, spec["variants"][resolved[name]])
    return graph


def _drop_node(graph: dict, node: str) -> None:
    """Remove a node and every input that was linked to it."""
    key = _graph_key(graph, node)
    del graph[key]
    for other in graph.values():
        for name in [n for n, v in other["inputs"].items() if _connected(v) and str(v[0]) == node]:
            del other["inputs"][name]
