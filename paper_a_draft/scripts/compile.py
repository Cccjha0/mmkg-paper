"""Compile the self-contained LaTeX draft with pdfLaTeX and BibTeX."""
import shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'.build'
BUILD.mkdir(exist_ok=True)
latex=shutil.which('pdflatex')
bib=shutil.which('bibtex')
if not latex or not bib:
    raise SystemExit('Install TeX Live or MiKTeX with pdflatex and bibtex, or upload this folder to Overleaf.')
for i,args in enumerate([
    [latex,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory=.build','main.tex'],
    [bib,'.build/main'],
    [latex,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory=.build','main.tex'],
    [latex,'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory=.build','main.tex'],
]):
    res=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output=res.stdout.decode('utf-8',errors='replace')
    (BUILD/f'compile_{i+1}.txt').write_text(output,encoding='utf-8')
    if res.returncode:
        print(output[-6500:])
        raise SystemExit(res.returncode)
shutil.copy2(BUILD/'main.pdf',ROOT/'main.pdf')
shutil.copy2(BUILD/'main.bbl',ROOT/'main.bbl')
log=(BUILD/'main.log').read_text(encoding='utf-8',errors='replace')
warnings=[line for line in log.splitlines() if any(s in line for s in ['Overfull','undefined','Warning:'])]
print('\n'.join(warnings) or 'Compilation clean: no overfull boxes or unresolved references.')
print('Compiled '+str(ROOT/'main.pdf'))
