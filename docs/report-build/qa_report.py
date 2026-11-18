import json
import re
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader

root = Path(__file__).resolve().parent
pdf = Path(sys.argv[1])
reader = PdfReader(pdf)
texts = [p.extract_text() or '' for p in reader.pages]
def compact(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s))

entries = json.loads((root / 'toc-entries.json').read_text(encoding='utf-8'))
first_body = next(i for i, text in enumerate(texts) if 'KẾ HOẠCH ĐỀ TÀI CỜ VUA AI' in text)
pages = {}
for entry in entries:
    needle = compact(entry['Title'])
    hits = [i for i in range(first_body, len(texts)) if needle in compact(texts[i])]
    if len(hits) != 1:
        raise ValueError(f'Expected one heading: {entry["Title"]}: {hits}')
    pages[entry['Bookmark']] = str(hits[0] - first_body + 1)
(root / 'page-map.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')

document = root.parent / 'Kế hoạch đề tài và phân công nhóm Cờ Vua AI.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
summary = {'pages': len(texts), 'body_pages': len(texts) - first_body, 'headings': len(entries)}
with ZipFile(document) as z:
    xmls = {n: etree.fromstring(z.read(n)) for n in z.namelist() if n.startswith('word/') and n.endswith('.xml')}
    fonts = set()
    colors = set()
    fills = set()
    for xml in xmls.values():
        for f in xml.xpath('//w:rFonts', namespaces=ns):
            for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
                val = f.get('{%s}%s' % (ns['w'], attr))
                if val:
                    fonts.add(val)
        colors.update(xml.xpath('//w:color/@w:val', namespaces=ns))
        fills.update(xml.xpath('//w:shd/@w:fill', namespaces=ns))
    assert fonts == {'Times New Roman'}, fonts
    assert colors == {'000000'}, colors
    assert not fills or fills <= {'FFFFFF', 'auto'}, fills
    xml = xmls['word/document.xml']
    assert len(xml.xpath('//w:tbl', namespaces=ns)) == 11
    assert len(xml.xpath('//w:tbl[1]/w:tr', namespaces=ns)) == 5
    assert len(xml.xpath('//w:bookmarkStart', namespaces=ns)) == len(entries)
    assert len(xml.xpath('//w:sectPr', namespaces=ns)) == 3
    assert xml[-1][-1].tag == '{%s}sectPr' % ns['w']
    if '--final' in sys.argv:
        toc = ''.join(xml.xpath('//w:p[w:pPr/w:pStyle[@w:val="TOC1" or @w:val="TOC2"]]//w:t/text()', namespaces=ns))
        for entry in entries:
            assert entry['Title'] + pages[entry['Bookmark']] in toc, entry
        assert not re.search(r'\.{3,}\s*0\b', texts[1])
    summary.update(fonts=sorted(fonts), colors=sorted(colors), fills=sorted(fills), tables=11, sections=3)
summary['page_words'] = [len(t.split()) for t in texts]
(root / 'qa-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('TOC page map:', pages)
