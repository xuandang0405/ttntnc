"""Check report fields and map rendered headings to printed page numbers."""
import json
import re
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from pypdf import PdfReader

root = Path(__file__).resolve().parent
workspace = root.parents[2]
document = workspace / 'docs/Báo cáo triển khai và huấn luyện các mô hình Cờ Vua AI.docx'
config = json.loads((root / 'report.json').read_text(encoding='utf-8'))
entries = json.loads((root / 'toc-entries.json').read_text(encoding='utf-8'))
texts = [p.extract_text() or '' for p in PdfReader(sys.argv[1]).pages]


def compact(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


first_body = next(i for i, text in enumerate(texts)
                  if compact(config['running_header']) in compact(text))
pages = {}
for entry in entries:
    hits = [i for i in range(first_body, len(texts))
            if compact(entry['Title']) in compact(texts[i])]
    if len(hits) != 1:
        raise ValueError(f'Expected one heading: {entry["Title"]}: {hits}')
    pages[entry['Bookmark']] = str(hits[0] - first_body + 1)
(root / 'page-map.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')

ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
summary = {'pages': len(texts), 'body_pages': len(texts) - first_body,
           'headings': len(entries), 'page_words': [len(t.split()) for t in texts]}
with ZipFile(document) as archive:
    xmls = [etree.fromstring(archive.read(name)) for name in archive.namelist()
            if name.startswith('word/') and name.endswith('.xml')]
    fonts, colors, fills = set(), set(), set()
    for xml in xmls:
        for font in xml.xpath('//w:rFonts', namespaces=ns):
            for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
                value = font.get(f'{{{ns["w"]}}}{attr}')
                if value:
                    fonts.add(value)
        colors.update(xml.xpath('//w:color/@w:val', namespaces=ns))
        fills.update(xml.xpath('//w:shd/@w:fill', namespaces=ns))
    assert fonts == {'Times New Roman'}, fonts
    assert colors == {'000000'}, colors
    assert not fills or fills <= {'FFFFFF', 'auto'}, fills
    xml = etree.fromstring(archive.read('word/document.xml'))
    tables = len(xml.xpath('//w:tbl', namespaces=ns))
    assert tables == 5, tables
    assert len(xml.xpath('//w:tbl[1]/w:tr', namespaces=ns)) == 5
    assert len(xml.xpath('//w:bookmarkStart', namespaces=ns)) == len(entries)
    assert len(xml.xpath('//w:sectPr', namespaces=ns)) == 3
    if '--final' in sys.argv:
        toc = ''.join(xml.xpath('//w:p[w:pPr/w:pStyle[@w:val="TOC1" or @w:val="TOC2"]]//w:t/text()', namespaces=ns))
        for entry in entries:
            assert entry['Title'] + pages[entry['Bookmark']] in toc, entry
    summary.update(fonts=sorted(fonts), colors=sorted(colors), fills=sorted(fills), tables=tables)
(root / 'qa-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
for index, text in enumerate(texts):
    body = text.replace(config['running_header'], '').strip().splitlines()
    print(f'PAGE {index + 1}: {body[:2]} ... {body[-3:]}')
