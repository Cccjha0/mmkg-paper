"""Check local dependencies and numerical inputs, render all PDF pages for QA."""
import csv, hashlib, json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import pymupdf as fitz
from PIL import Image, ImageOps, ImageDraw

main=(ROOT/'main.tex').read_text(encoding='utf-8')
# Only active inputs belong to the current draft; historical tables remain archived.
tex_files=[ROOT/'main.tex']+[ROOT/rel for rel in re.findall(r'\\input\{([^}]+)\}',main)]
source='\n'.join(p.read_text(encoding='utf-8') for p in tex_files)
main=(ROOT/'main.tex').read_text(encoding='utf-8')
for rel in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',main):
    assert (ROOT/rel).is_file(),rel
labels=re.findall(r'\\label\{([^}]+)\}',source)
assert len(labels)==len(set(labels)), 'Duplicate label'
for key in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source): assert key in labels,key
bib=(ROOT/'references.bib').read_text(encoding='utf-8')
keys=set(re.findall(r'@\w+\{([^,]+),',bib))
for item in re.findall(r'\\cite\w*\{([^}]+)\}',main):
    assert set(item.split(','))<=keys,item
manifest=json.loads((ROOT/'rerun_source_manifest.json').read_text(encoding='utf-8'))
assert manifest['version']=='information_boundary_v2'
for rel,sha in manifest['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for name,sha in manifest['tables'].items():
    assert hashlib.sha256((ROOT/'tables/rerun'/name).read_bytes()).hexdigest()==sha,name
dyna=json.loads((ROOT/'dynasemble_source_manifest.json').read_text(encoding='utf-8'))
assert dyna['version']=='dynasemble_controls_v1_review'
for rel,sha in dyna['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for rel,sha in dyna['tables'].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables={ROOT/'tables/rerun'/name for name in manifest['tables']}|{ROOT/rel for rel in dyna['tables']}
conservative=json.loads((ROOT/'conservative_source_manifest.json').read_text(encoding='utf-8'))
assert conservative['version']=='conservative_radius_review_v1'
for rel,sha in conservative['sources'].items():
    assert hashlib.sha256((ROOT.parent/rel).read_bytes()).hexdigest()==sha,rel
for collection in ('tables','figures'):
    for rel,sha in conservative[collection].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
bound_tables|={ROOT/rel for rel in conservative['tables']}
assert set(tex_files[1:])<=bound_tables
radius_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/conservative_radius_review/audit.json').read_text())
assert radius_audit['status']=='conservative_radius_checks_passed' and not radius_audit['failures']
assert radius_audit['test_used_for_new_selection'] is False
assert len(radius_audit['pair_checks'])==4
assert all(c['five_fold_actions_rr_fallback_reproduced'] for c in radius_audit['pair_checks'])
dyna_audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/dynasemble_review_audit/audit.json').read_text())
assert dyna_audit['status']=='review_artifact_checks_passed' and not dyna_audit['failures']
assert dyna_audit['small_cache_evidence_verified'], 'D01-D03 closure requires verified cache identities and selector replay'
audit=json.loads((ROOT.parent/'outputs/paper_a_safe_correction/information_boundary_rerun_audit/audit.json').read_text())
assert audit['status']=='artifact_checks_passed_with_continuous_repeatability_caveat'
assert not audit['failures']
build=ROOT/'.build/qa'; build.mkdir(exist_ok=True,parents=True)
doc=fitz.open(ROOT/'main.pdf')
extracted='\n'.join(p.get_text() for p in doc)
assert '??' not in extracted,'Unresolved reference in PDF'
assert not re.search(r'\b(?:TODO|TBD|PLACEHOLDER)\b',extracted),'Placeholder found'
page_info=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False)
    pix.save(build/f'page_{i+1:02d}.png')
    page_info.append({'page':i+1,'characters':len(page.get_text()),'images':len(page.get_images())})
    for block in page.get_text('dict')['blocks']:
        if block['type']!=0: continue
        x0,y0,x1,y1=block['bbox']
        assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(i+1,block['bbox'])
for start in range(0,len(doc),6):
    sheet=Image.new('RGB',(3*398,2*588),'#d9dde0')
    draw=ImageDraw.Draw(sheet)
    for offset,i in enumerate(range(start,min(start+6,len(doc)))):
        im=Image.open(build/f'page_{i+1:02d}.png'); im.thumbnail((378,548))
        x=(offset%3)*398+10; y=(offset//3)*588+26
        sheet.paste(im,(x,y)); draw.text((x,y-20),f'Page {i+1}',fill='black')
    sheet.save(build/f'contact_{start//6+1}.png')
report={'pages':len(doc),'bibliography_entries':len(keys),'tables':len(re.findall(r'\\begin\{table\}',source)),
        'figures':len(re.findall(r'\\begin\{figure\}',source)),'source_snapshots':len(set(manifest['sources'])|set(dyna['sources'])|set(conservative['sources'])),
        'result_version':'information_boundary_v2 + dynasemble_controls_v1_review + conservative_radius_review_v1',
        'small_cache_evidence_verified':dyna_audit['small_cache_evidence_verified'],
        'conservative_radius_checks_passed':True,'test_used_for_new_selection':False,
        'checks':'active input paths, references, rerun source/table hashes, audit status, no placeholders, page bounds: PASS',
        'page_details':page_info}
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(build/'extracted_text.txt').write_text(extracted,encoding='utf-8')
print(json.dumps(report,indent=2))
