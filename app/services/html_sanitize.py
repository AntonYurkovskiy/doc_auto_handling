"""Безопасное отображение сохранённого HTML-письма заявки.

Письмо показываем «как пришло», но без активного содержимого: удаляем скрипты,
стили-ссылки, фреймы, обработчики событий и любые внешние ресурсы. Результат
рассчитан на вывод внутри `<iframe sandbox srcdoc="...">`, то есть даже при
пропущенном теге браузер не выполнит скрипт и не отправит форму.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment, Tag

# Разрешена только разметка текста и таблиц — того, из чего состоят письма-заявки.
ALLOWED_TAGS = {
    "a", "b", "blockquote", "br", "caption", "col", "colgroup", "div", "em", "font",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "li", "ol", "p", "pre", "s",
    "small", "span", "strong", "sub", "sup", "table", "tbody", "td", "tfoot", "th",
    "thead", "tr", "u", "ul",
}
# Теги, удаляемые вместе с содержимым (остальные неизвестные — «разворачиваются»).
DROPPED_TAGS = {
    "applet", "audio", "base", "button", "embed", "form", "frame", "frameset",
    "iframe", "input", "link", "math", "meta", "noscript", "object", "script",
    "select", "source", "style", "svg", "textarea", "track", "video",
}
ALLOWED_ATTRS = {
    "align", "bgcolor", "border", "cellpadding", "cellspacing", "class", "color",
    "colspan", "dir", "height", "rowspan", "size", "style", "title", "valign", "width",
}
_SAFE_HREF = re.compile(r"^(?:https?:|mailto:|#)", re.IGNORECASE)
_UNSAFE_STYLE = re.compile(r"(?:url\s*\(|expression\s*\(|javascript:|@import)", re.IGNORECASE)


def sanitize_email_html(html: str | None) -> str:
    """Вернуть HTML письма без скриптов, внешних ресурсов и обработчиков событий."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    for tag in list(soup.find_all(True)):
        if not isinstance(tag, Tag) or tag.decomposed:
            continue
        name = tag.name.lower()
        if name in DROPPED_TAGS:
            tag.decompose()
            continue
        if name == "img":
            # Внешние картинки не подгружаем: заменяем на текстовую отметку.
            tag.replace_with(f"[изображение: {tag.get('alt') or tag.get('src') or ''}]".strip())
            continue
        if name not in ALLOWED_TAGS:
            tag.unwrap()
            continue
        tag.attrs = _clean_attrs(name, tag.attrs)

    return soup.decode()


def _clean_attrs(tag_name: str, attrs: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for attr, value in attrs.items():
        low = attr.lower()
        if low.startswith("on") or low in {"src", "srcset", "background", "formaction"}:
            continue
        if low == "style" and isinstance(value, str) and _UNSAFE_STYLE.search(value):
            continue
        if tag_name == "a" and low == "href":
            if isinstance(value, str) and _SAFE_HREF.match(value.strip()):
                cleaned[attr] = value
            continue
        if low in ALLOWED_ATTRS:
            cleaned[attr] = value
    if tag_name == "a" and "href" in cleaned:
        cleaned["rel"] = "noopener noreferrer nofollow"
        cleaned["target"] = "_blank"
    return cleaned
