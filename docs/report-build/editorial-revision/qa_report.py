"""Verify cover fidelity, editorial constraints, numbering and rendered TOC."""
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
source = workspace / 'docs/Kế hoạch đề tài Cờ Vua AI - Hoàn thiện.docx'
output = workspace / 'docs/Kế hoạch đề tài Cờ Vua AI - Bản biên tập.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
entries = json.loads((root / 'toc-entries.json').read_text(encoding='utf-8'))
manuscript = json.loads((root / 'report.json').read_text(encoding='utf-8'))


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=ns))


def compact(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value))


def cover_nodes(xml):
    nodes = list(xml.find('w:body', ns))
    index = next(i for i, node in enumerate(nodes) if node.find('w:pPr/w:sectPr', ns) is not None)
    return nodes[:index + 1]


def canonical(data):
    return etree.tostring(etree.fromstring(data), method='c14n', exclusive=True)


with ZipFile(source) as before, ZipFile(output) as after:
    original = etree.fromstring(before.read('word/document.xml'))
    edited = etree.fromstring(after.read('word/document.xml'))
    old_cover, new_cover = cover_nodes(original), cover_nodes(edited)
    assert len(old_cover) == len(new_cover)
    for old, new in zip(old_cover, new_cover):
        assert etree.tostring(old, method='c14n', exclusive=True) == etree.tostring(new, method='c14n', exclusive=True)
    for name in before.namelist():
        if name.startswith('word/') and name != 'word/document.xml':
            old_part, new_part = before.read(name), after.read(name)
            if old_part != new_part and name.endswith('.xml'):
                old_part, new_part = canonical(old_part), canonical(new_part)
            assert old_part == new_part, name
    for entry in entries:
        bookmark = edited.xpath('//w:bookmarkStart[@w:name=$name]', namespaces=ns, name=entry['Bookmark'])[0]
        heading = bookmark.getparent()
        assert text(heading) == entry['Title']
        assert not heading.xpath('./w:pPr/w:numPr', namespaces=ns)
    references = edited.xpath('//w:p[w:pPr/w:pStyle[@w:val="Bibliography"]]', namespaces=ns)
    assert len(references) == 7
    fonts, colors, fills = set(), set(), set()
    for name in after.namelist():
        if not (name.startswith('word/') and name.endswith('.xml')):
            continue
        xml = etree.fromstring(after.read(name))
        for font in xml.xpath('//w:rFonts', namespaces=ns):
            for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
                value = font.get(f'{{{ns["w"]}}}{attr}')
                if value:
                    fonts.add(value)
        colors.update(xml.xpath('//w:color/@w:val', namespaces=ns))
        fills.update(xml.xpath('//w:shd/@w:fill', namespaces=ns))
    assert fonts == {'Times New Roman'} and colors == {'000000'} and not fills
    assert len(edited.xpath('//w:sectPr', namespaces=ns)) == 3
    assert len(edited.xpath('//w:tbl', namespaces=ns)) == 13
    body_text = text(edited)
    for unwanted in ('Đầu value', 'Policy head', 'planes20-v1', '5.120 byte', '1.536 byte', 'sức mạnh chơi', 'có tính mô tả', 'cấu hình teacher'):
        assert unwanted not in body_text, unwanted
    for member_id in ('3123410065', '3123410072', '3123410426', '3123410077'):
        assert body_text.count(member_id) >= 2
    for chapter in manuscript['chapters']:
        for section in chapter['sections']:
            if 'table' in section:
                table_number = re.match(r'Bảng \d+\.\d+', section['table']['caption']).group(0)
                assert table_number in ' '.join(section['paragraphs']), table_number

reader = PdfReader(sys.argv[1])
pages = [page.extract_text() or '' for page in reader.pages]
first_body = next(i for i, content in enumerate(pages)
                  if i > 0 and 'MỤC LỤC' not in content
                  and compact(entries[0]['Title']) in compact(content))
page_map = {}
for entry in entries:
    found = [i for i in range(first_body, len(pages)) if compact(entry['Title']) in compact(pages[i])]
    assert len(found) == 1, (entry['Title'], found)
    page_map[entry['Bookmark']] = str(found[0] - first_body + 1)
(root / 'page-map.json').write_text(json.dumps(page_map, indent=2), encoding='utf-8')
if '--final' in sys.argv:
    toc_xml = ''.join(text(p) for p in edited.xpath('//w:p[w:pPr/w:pStyle[@w:val="TOC1" or @w:val="TOC2"]]', namespaces=ns))
    rendered_toc = compact(re.sub(r'\.{2,}', '', '\n'.join(pages[1:first_body])))
    for entry in entries:
        expected = entry['Title'] + page_map[entry['Bookmark']]
        assert expected in toc_xml, entry
        assert compact(expected) in rendered_toc, entry
summary = dict(pages=len(pages), body_pages=len(pages) - first_body,
               page_words=[len(p.split()) for p in pages], cover_preserved=True,
               visible_headings=len(entries), tables=13, references=len(references),
               fonts=sorted(fonts), colors=sorted(colors))
(root / 'qa-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
for i, content in enumerate(pages):
    lines = [line for line in content.splitlines() if line.strip()]
    print(f'PAGE {i+1}: {lines[:2]} ... {lines[-3:]}')
