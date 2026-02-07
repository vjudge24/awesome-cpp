"""Tests for the prompt versioning registry."""

import json
import tempfile
from pathlib import Path

import pytest

from src.prompts.registry import PromptRegistry, PromptVersion, get_default_registry


class TestPromptVersion:
    def test_create_with_auto_hash(self):
        pv = PromptVersion(name="test", version="1.0.0", template="Hello {name}")
        assert pv.content_hash  # Auto-generated
        assert len(pv.content_hash) == 12

    def test_same_content_same_hash(self):
        pv1 = PromptVersion(name="a", version="1.0", template="Hello")
        pv2 = PromptVersion(name="b", version="2.0", template="Hello")
        assert pv1.content_hash == pv2.content_hash

    def test_different_content_different_hash(self):
        pv1 = PromptVersion(name="a", version="1.0", template="Hello")
        pv2 = PromptVersion(name="a", version="1.0", template="World")
        assert pv1.content_hash != pv2.content_hash

    def test_render(self):
        pv = PromptVersion(name="test", version="1.0", template="Hello {name}!")
        assert pv.render(name="World") == "Hello World!"

    def test_to_dict(self):
        pv = PromptVersion(name="test", version="1.0", template="tmpl")
        d = pv.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0"
        assert d["template"] == "tmpl"
        assert "content_hash" in d
        assert "created_at" in d


class TestPromptRegistry:
    def test_register_and_get(self):
        registry = PromptRegistry()
        pv = PromptVersion(name="router", version="1.0", template="Classify intent")
        registry.register(pv)

        result = registry.get("router")
        assert result.template == "Classify intent"

    def test_get_specific_version(self):
        registry = PromptRegistry()
        v1 = PromptVersion(name="p", version="1.0", template="v1")
        v2 = PromptVersion(name="p", version="2.0", template="v2")
        registry.register(v1)
        registry.register(v2)

        assert registry.get("p", version="1.0").template == "v1"
        assert registry.get("p", version="2.0").template == "v2"

    def test_get_latest_is_last_registered(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="p", version="1.0", template="v1"))
        registry.register(PromptVersion(name="p", version="2.0", template="v2"))

        assert registry.get("p").version == "2.0"

    def test_get_nonexistent_raises(self):
        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_nonexistent_version_raises(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="p", version="1.0", template="t"))
        with pytest.raises(KeyError):
            registry.get("p", version="9.9")

    def test_duplicate_registration_idempotent(self):
        registry = PromptRegistry()
        pv = PromptVersion(name="p", version="1.0", template="t")
        registry.register(pv)
        registry.register(pv)  # Should not raise
        assert len(registry.list_versions("p")) == 1

    def test_duplicate_version_different_content_raises(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="p", version="1.0", template="v1"))
        with pytest.raises(ValueError, match="already exists"):
            registry.register(PromptVersion(name="p", version="1.0", template="v2"))

    def test_list_versions(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="p", version="2.0", template="b"))
        registry.register(PromptVersion(name="p", version="1.0", template="a"))
        assert registry.list_versions("p") == ["1.0", "2.0"]

    def test_list_versions_nonexistent(self):
        registry = PromptRegistry()
        assert registry.list_versions("x") == []

    def test_list_prompts(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="a", version="1.0", template="ta"))
        registry.register(PromptVersion(name="b", version="2.0", template="tb"))
        prompts = registry.list_prompts()
        assert prompts == {"a": "1.0", "b": "2.0"}

    def test_save_and_load(self):
        registry = PromptRegistry()
        registry.register(PromptVersion(name="p1", version="1.0", template="hello"))
        registry.register(PromptVersion(name="p1", version="2.0", template="world"))
        registry.register(PromptVersion(name="p2", version="1.0", template="foo"))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        registry.save(path)

        # Load into new registry
        new_registry = PromptRegistry()
        new_registry.load(path)

        assert new_registry.get("p1", "1.0").template == "hello"
        assert new_registry.get("p1", "2.0").template == "world"
        assert new_registry.get("p2").template == "foo"
        Path(path).unlink()


class TestDefaultRegistry:
    def test_default_registry_has_required_prompts(self):
        registry = get_default_registry()
        prompts = registry.list_prompts()
        assert "router_system" in prompts
        assert "synthesizer_system" in prompts
        assert "quality_check" in prompts
        assert "query_decomposition" in prompts

    def test_default_prompts_have_content(self):
        registry = get_default_registry()
        for name in registry.list_prompts():
            prompt = registry.get(name)
            assert prompt.template
            assert prompt.content_hash
