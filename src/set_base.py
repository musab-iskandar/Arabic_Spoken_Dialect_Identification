#!/usr/bin/env python
"""
Stamp the public URL of the site into the files that need an absolute one.

    python src/set_base.py https://asdi-nadi2026.surge.sh
    python src/set_base.py https://user.github.io/Repo        # sub-path host

Canonical links, Open Graph images and sitemap entries cannot be relative --
scrapers fetch them out of context, so they have to carry the real origin. This
stamps that origin into:

    site/index.html    canonical, og:url, og:image, twitter:image
    site/robots.txt    Sitemap:
    site/sitemap.xml   <loc>
    site/404.html      the icon and "back" links

Everything else on the page is already relative and needs no change.

WHY 404.html IS TREATED SEPARATELY
----------------------------------
Static hosts serve 404.html for ANY path that does not exist, so a relative link
inside it resolves against whatever the visitor typed -- /a/b/c/ gives you
/a/b/c/favicon.svg, which is not there. Its links therefore need an absolute
path. On a root host that is just "/"; on a project host such as GitHub Pages it
is "/RepoName". Both are derived from the URL passed in.

The script reads the CURRENT base out of the canonical tag, so it is idempotent
and can be re-run to move the site between hosts.
"""

import io
import os
import re
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, 'site')
PLACEHOLDER = '__BASE__'


def current_base():
    """Whatever is stamped now, read off the canonical tag."""
    s = io.open(os.path.join(SITE, 'index.html'), encoding='utf-8').read()
    m = re.search(r'<link rel="canonical" href="([^"]+)"', s)
    if not m:
        return None
    return m.group(1).rstrip('/')


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"current base: {current_base() or '(none)'}")
        sys.exit(1)

    new = sys.argv[1].rstrip('/')
    u = urlparse(new)
    if u.scheme not in ('http', 'https') or not u.netloc:
        print(f"FAIL: '{new}' is not an absolute http(s) URL.")
        sys.exit(2)
    prefix = u.path.rstrip('/') or ''          # '' at a domain root, '/Repo' on a project host

    old = current_base()
    if old is None:
        print('FAIL: no canonical tag in site/index.html -- cannot tell what to replace.')
        sys.exit(2)

    old_prefix = urlparse(old).path.rstrip('/') if old != PLACEHOLDER else ''
    print(f'  from : {old}')
    print(f'  to   : {new}')
    print(f'  path prefix for 404 links: {prefix or "/"}\n')

    total = 0
    for rel in ('index.html', 'robots.txt', 'sitemap.xml'):
        p = os.path.join(SITE, rel)
        if not os.path.exists(p):
            print(f'  {rel:<14s} missing, skipped')
            continue
        s = io.open(p, encoding='utf-8').read()
        n = s.count(old) + s.count(PLACEHOLDER)
        s = s.replace(old, new).replace(PLACEHOLDER, new)
        io.open(p, 'w', encoding='utf-8').write(s)
        total += n
        print(f'  {rel:<14s} {n} replaced')

    p = os.path.join(SITE, '404.html')
    if os.path.exists(p):
        s = io.open(p, encoding='utf-8').read()
        before = s
        for target in ('favicon.svg', ''):
            was = f'{old_prefix}/{target}' if old_prefix else f'/{target}'
            now = f'{prefix}/{target}' if prefix else f'/{target}'
            s = s.replace(f'href="{was}"', f'href="{now}"')
        io.open(p, 'w', encoding='utf-8').write(s)
        print(f'  {"404.html":<14s} links {"updated" if s != before else "already correct"}')

    print(f'\n{total} absolute references now point at {new}')
    print('Re-run this any time the site moves; it reads the current value itself.')


if __name__ == '__main__':
    main()
