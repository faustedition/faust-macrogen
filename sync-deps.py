#!/usr/bin/env python3

"""
This script reads the dependencies from poetry's section in pyproject.toml and
updates environment.yml accordingly.

Dependencies missing from environment.yml will only be added if they are not
marked as optional. Optional dependencies that have been added to environment.yml
manually will be updated, however.
"""

from typing import Mapping

import toml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
import re


def load_poetry_deps():
    with open("pyproject.toml") as f:
        pyproject = toml.load(f)
    return pyproject["tool"]["poetry"]["dependencies"]


def load_conda_deps():
    with open("environment.yml") as f:
        env = yaml.load(f)
        env_dep_list: CommentedSeq = env["dependencies"]
        env_deps = {}
        for dep_idx, dep_spec in enumerate(env_dep_list):
            match = re.match(r"^([-\w.]+)\s*([<>=].*?)\s*$", dep_spec)
            if match:
                env_deps[match.group(1)] = (
                    dep_idx,
                    match.start(2),
                    match.end(2),
                    match.group(2),
                )
            else:
                print("Could not parse", dep_spec, "in environment.yml")
    return env, env_dep_list, env_deps


def poetry2condaspec(ver):
    if ver[0] == "^":
        lower_bound = ver[1:]
        parts = lower_bound.split(".")
        if parts[0] == "0":
            upper_bound = ".".join(["0", str(int(parts[1]) + 1)])
        else:
            upper_bound = str(int(parts[0]) + 1) + ".0"
        new_env_ver = f">={lower_bound},<{upper_bound}"
    else:
        new_env_ver = ver
    return new_env_ver


def main():
    poetry_deps = load_poetry_deps()
    env, env_dep_list, env_deps = load_conda_deps()

    for lib, ver in poetry_deps.items():
        if isinstance(ver, Mapping):
            optional = ver.get("optional", False)
            ver = ver.get("version", "")
        else:
            optional = False

        new_env_ver = poetry2condaspec(ver)

        if lib in env_deps:
            idx, start, end, old_spec = env_deps[lib]
            if old_spec != new_env_ver:
                env_dep_list[idx] = lib + new_env_ver
            if ver != new_env_ver:
                env_dep_list.yaml_add_eol_comment(ver, idx, 40)
            if old_spec != new_env_ver:
                print(f"{lib:>30}: {old_spec:20} => {new_env_ver:20} ({ver})")
        elif not optional:
            env_dep_list.append(lib + new_env_ver)
            env_dep_list.yaml_add_eol_comment("new", len(env_dep_list) - 1, 40)
            print(f"{lib:>30}: new:                    {new_env_ver:20} ({ver})")

    with open("environment.yml", "w") as f:
        yaml.dump(env, f)


if __name__ == "__main__":
    main()
