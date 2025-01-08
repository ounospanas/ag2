# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
from pathlib import Path
import shutil
import os

# Add the ../../website directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'website'))
from process_api_reference import move_files_excluding_index


def create_test_directory_structure(tmp_path):
    """Helper function to create test directory structure"""
    # Create autogen directory
    autogen_dir = tmp_path / "autogen"
    autogen_dir.mkdir()

    # Create some files in autogen
    (autogen_dir / "browser_utils.md").write_text("browser_utils")
    (autogen_dir / "code_utils.md").write_text("code_utils")

    # Create subdirectories with files
    subdir1 = autogen_dir / "agentchat"
    subdir1.mkdir()
    (subdir1 / "assistant_agent.md").write_text("assistant_agent")
    (subdir1 / "conversable_agent.md").write_text("conversable_agent")
    (subdir1 / "index.md").write_text("index")

    # nested subdirectory
    nested_subdir = subdir1 / "contrib"
    nested_subdir.mkdir()
    (nested_subdir / "agent_builder.md").write_text("agent_builder")
    (nested_subdir / "index.md").write_text("index")

    subdir2 = autogen_dir / "cache"
    subdir2.mkdir()
    (subdir2 / "cache_factory.md").write_text("cache_factory")
    (subdir2 / "disk_cache.md").write_text("disk_cache")
    (subdir2 / "index.md").write_text("index")

    return tmp_path

def test_move_files_excluding_index(tmp_path):
    """Test that files are moved correctly excluding index.md"""
    # Setup the test directory structure
    api_dir = create_test_directory_structure(tmp_path)
    
    # Call the function under test
    move_files_excluding_index(api_dir)

    # Verify that autogen directory no longer exists
    assert not (api_dir / "autogen").exists()

    # Verify files were moved correctly
    assert (api_dir / "browser_utils.md").exists()
    assert (api_dir / "code_utils.md").exists()
    assert (api_dir / "agentchat" / "assistant_agent.md").exists()
    assert (api_dir / "agentchat" / "conversable_agent.md").exists()
    assert (api_dir / "agentchat" / "contrib" / "agent_builder.md").exists()
    assert (api_dir / "cache" / "cache_factory.md").exists()
    assert (api_dir / "cache" / "disk_cache.md").exists()

    # Verify index.md was not moved
    assert not (api_dir / "agentchat" / "index.md").exists()
    assert not (api_dir / "agentchat" / "contrib" / "index.md").exists()
    assert not (api_dir / "cache" / "index.md").exists()

