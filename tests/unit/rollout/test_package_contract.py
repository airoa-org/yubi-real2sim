# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Architecture contracts for the extracted rollout package."""

from tests._helpers.worker_ast import *  # noqa: F403

def test_rollout_modules_have_no_import_cycles() -> None:
    """Extracted rollout modules must remain a directed acyclic graph."""

    module_paths = {
        str(path.relative_to(REPO_ROOT)).removesuffix(".py").replace("/", ".")
        for path in _WORKER_SOURCE_FILES
    }
    graph = {module: set() for module in module_paths}
    for path, tree in _WORKER_TREES.items():
        module = str(path.relative_to(REPO_ROOT)).removesuffix(".py").replace("/", ".")
        for node in ast.walk(tree):
            imported: str | None = None
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    package_parts = module.split(".")[:-node.level]
                    imported = ".".join((*package_parts, node.module))
                else:
                    imported = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_paths:
                        graph[module].add(alias.name)
                continue
            if imported in module_paths:
                graph[module].add(imported)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str) -> None:
        assert module not in visiting, f"rollout import cycle at {module}"
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)

def test_no_module_globals_in_rollout_modules() -> None:
    """Rollout state must be explicit rather than injected by ``global``."""

    violations = [
        str(path.relative_to(REPO_ROOT))
        for path, tree in _WORKER_TREES.items()
        if any(isinstance(node, ast.Global) for node in ast.walk(tree))
    ]
    assert violations == []

def test_rollout_modules_keep_heavy_imports_lazy() -> None:
    """Optional scene, policy, and wrist backends stay below function scope."""

    forbidden = {
        "jax",
        "openpi",
        "pandas",
        "scene.gs_background_client",
        "robosuite_env.policy.policy_batch_infer",
        "robosuite_env.policy.policy_batch_server",
        "robosuite_env.policy.rel2abs",
        "robosuite_env.rendering.wrist_fisheye",
        "robosuite_env.rollout.sources.lerobot",
    }
    violations: list[tuple[str, str]] = []
    for path, tree in _WORKER_TREES.items():
        for node in tree.body:
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if any(
                    module == name or module.startswith(name + ".")
                    for name in forbidden
                ):
                    violations.append((str(path.relative_to(REPO_ROOT)), module))
    assert violations == []
