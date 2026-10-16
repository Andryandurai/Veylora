"""The site logo: web copy for the sidebar, square icon for the browser tab, and the text fallback."""

from __future__ import annotations

import os

import pytest
from PIL import Image

from dashboards.components import assets


@pytest.fixture
def logo_files(tmp_path, monkeypatch):
    source = tmp_path / "logo.png"
    target = tmp_path / "static" / "veylora-logo.jpg"
    monkeypatch.setattr(assets, "LOGO_SOURCE", source)
    monkeypatch.setattr(assets, "LOGO_TARGET", target)
    assets.favicon.cache_clear()
    yield source, target
    assets.favicon.cache_clear()


def _write_logo(path, size=(1254, 1254)):
    Image.new("RGB", size, (6, 26, 22)).save(path)


def test_without_a_logo_file_there_is_no_logo_and_no_icon(logo_files):
    assert assets.ensure_logo() is None
    assert assets.favicon() is None


def test_logo_is_downscaled_for_the_web_and_refreshed_when_the_source_changes(logo_files):
    source, target = logo_files
    _write_logo(source)
    assert assets.ensure_logo() == assets.LOGO_URL
    with Image.open(target) as web:
        assert web.size == (assets.LOGO_WEB_PX, assets.LOGO_WEB_PX)
        assert web.format == "JPEG"
    # a newer source replaces the web copy
    Image.new("RGB", (800, 800), (200, 10, 10)).save(source)
    target_mtime = target.stat().st_mtime
    os.utime(source, (target_mtime + 10, target_mtime + 10))
    assets.ensure_logo()
    with Image.open(target) as web:
        assert web.size == (assets.LOGO_WEB_PX, assets.LOGO_WEB_PX)
        assert web.getpixel((10, 10))[0] > 150


def test_favicon_is_a_small_square_cut_from_the_emblem(logo_files):
    source, _ = logo_files
    _write_logo(source, size=(1000, 1000))
    icon = assets.favicon()
    assert icon is not None and icon.size == (assets.ICON_PX, assets.ICON_PX)
    # the padding uses the logo's own background colour, so the icon has no white border
    assert icon.getpixel((0, 0)) == (6, 26, 22)


def test_removed_logo_stops_being_served_even_if_a_web_copy_is_left_behind(logo_files):
    source, target = logo_files
    _write_logo(source)
    assert assets.ensure_logo() == assets.LOGO_URL
    source.unlink()
    assert assets.ensure_logo() is None
    assert target.exists()
