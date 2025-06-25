# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import tempfile
import textwrap
from pathlib import Path

import pytest

from autogen.mcp.helpers import run_streamable_http_client  # Adjust import path


@pytest.mark.asyncio
async def test_run_streamable_http_client_lifecycle():
    # Create a temporary dummy server script that simulates "streamable-http"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as dummy_script:
        dummy_script.write(
            textwrap.dedent("""
            import time
            import sys
            import signal

            def handle_sigint(signum, frame):
                print("Received SIGINT", flush=True)
                sys.exit(0)

            signal.signal(signal.SIGINT, handle_sigint)

            if sys.argv[1] == "streamable-http":
                print("Server starting", flush=True)
                while True:
                    time.sleep(1)
        """)
        )
        dummy_script.close()
        script_path = dummy_script.name

        try:
            async with run_streamable_http_client(mcp_server_path=script_path, startup_wait_secs=1.0) as proc:
                assert proc.returncode is None, "Process should be running"
                # Ensure it prints something to stdout (like "Server starting")
                stdout = await asyncio.wait_for(proc.stdout.readline(), timeout=2.0)
                assert b"Server starting" in stdout
            # After context manager, process should be terminated
            assert proc.returncode is not None, "Process should be terminated after context manager"
        finally:
            Path(script_path).unlink()  # Clean up
