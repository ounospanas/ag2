# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from .patch_fastapi_code_generator import (  # noqa: E402
    patch_function_name_parsing,
    patch_generate_code,
)

patch_function_name_parsing()
patch_generate_code()

from .mcp_proxy import MCPProxy  # noqa: E402

__all__ = ["MCPProxy"]
