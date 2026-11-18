"""Verify the cover edit, member fields, preserved content and TOC pages."""
import json
import re
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from pypdf import PdfReader

workspace = Path(__file__).resolve().parents[3]
source = workspace / 'docs/Kế hoạch đề tài Cờ Vua AI - Bản chỉnh sửa.docx'
output = workspace / 'docs/Kế hoạch đề tài Cờ Vua AI - Bản nộp.docx'
pdf = Path(sys.argv[1])
root = Path(__file__).resolve().parent
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
members = [('Phạm Minh Dương', '3123410065'), ('Phạm Tấn Đạt', '3123410072'),
           ('Đỗ Quốc Việt', '3123410426'), ('Nguyễn Xuân Đăng', '3123410077')]


def compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def body_after_cover(xml):
    body = xml.find('w:body', ns)
    index = next(i for i, node in enumerate(body) if node.find('w:pPr/w:sectPr', ns) is not None)
    return list(body)[index + 1:]


def plain(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=ns))


with ZipFile(source) as before, ZipFile(output) as after:
    original = etree.fromstring(before.read('word/document.xml'))
    edited = etree.fromstring(after.read('word/document.xml'))
    old_blocks, new_blocks = body_after_cover(original), body_after_cover(edited)
    assert len(old_blocks) == len(new_blocks)
    member_index = 0
    for old, new in zip(old_blocks, new_blocks):
        if plain(old).startswith('Họ tên:'):
            name, student_id = members[member_index]
            assert plain(new) == f'Họ tên: {name}    Mã số sinh viên: {student_id}'
            member_index += 1
            for old_text, new_text in zip(old.xpath('.//w:t', namespaces=ns), new.xpath('.//w:t', namespaces=ns)):
                new_text.text = old_text.text
        assert etree.tostring(old, method='c14n', exclusive=True) == etree.tostring(new, method='c14n', exclusive=True), plain(old)
    assert member_index == 4
    for name in before.namelist():
        if name.startswith('word/') and name != 'word/document.xml' and not name.endswith('.rels'):
            assert before.read(name) == after.read(name), name
    xmls = [etree.fromstring(after.read(name)) for name in after.namelist()
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
    assert not fills, fills
    assert len(edited.xpath('//w:sectPr', namespaces=ns)) == 3

pages = [page.extract_text() or '' for page in PdfReader(pdf).pages]
assert len(pages) == 21, len(pages)
cover = pages[0]
for expected in ['TRƯỜNG ĐẠI HỌC SÀI GÒN', 'KHOA CÔNG NGHỆ THÔNG TIN',
                 'BÁO CÁO KẾ HOẠCH ĐỀ TÀI', 'TRÍ TUỆ NHÂN TẠO NÂNG CAO', 'Đỗ Như Tài']:
    assert compact(expected) in compact(cover), expected
for name, student_id in members:
    assert compact(name) in compact(cover) and student_id in cover
assert 'Phạm Thi Vương' not in ''.join(pages)
assert 'chatbot' not in cover.lower()
assert not re.search(r'\.{8,}', ''.join(pages[2:]))

entries = json.loads((root.parent / 'revision-v2/toc-entries.json').read_text(encoding='utf-8'))
actual_pages = {}
for entry in entries:
    matches = [i for i in range(2, len(pages)) if compact(entry['Title']) in compact(pages[i])]
    assert len(matches) == 1, (entry['Title'], matches)
    number = str(matches[0] - 1)
    assert compact(entry['Title'] + number) in compact(re.sub(r'\.{2,}', '', pages[1])), entry
    actual_pages[entry['Bookmark']] = number
summary = dict(pages=len(pages), members=members, fonts=sorted(fonts), colors=sorted(colors),
               content_preserved=True, member_fields_filled=member_index,
               toc_entries_verified=len(entries), tables=len(edited.xpath('//w:tbl', namespaces=ns)))
(root / 'qa-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
