"""Flatten the report and collect only the vector images needed by Overleaf."""

from pathlib import Path
import re
import shutil
import zipfile

HERE=Path(__file__).resolve().parent


def expand(source):
    def replace(match):
        name=match.group(1)
        path=HERE/(name if name.endswith('.tex') else name+'.tex')
        return '\n% Included from '+path.name+'\n'+expand(path.read_text())+'\n'
    source=re.sub(r'\\input\{([^}]+)\}',replace,source)
    source=re.sub(r'\\tableinput\s+([\w_]+\.tex)\s*',lambda m:(HERE/m.group(1)).read_text()+'\n',source)
    return source


def main():
    destination=HERE/'overleaf'
    pictures=destination/'Bilder'
    pictures.mkdir(parents=True,exist_ok=True)
    source=expand((HERE/'bericht.tex').read_text())
    source=source.replace(r'\graphicspath{{figures/}}',r'\graphicspath{{Bilder/}}')
    source=source.replace(r'\makeatletter\let\tableinput\@@input\makeatother','')
    # Standard LaTeX, no Python, shell escape, auxiliary .tex files or CSV
    # readers needed in Overleaf. Measurements/tables are embedded literally.
    assert r'\input{' not in source and r'\tableinput ' not in source
    (destination/'report.tex').write_text(source)
    needed=set(re.findall(r'(?<![\w/])([0-9]{2}_[\w]+\.pdf)',source))
    for name in needed:
        shutil.copy2(HERE/'figures'/name,pictures/name)
    instructions=('Overleaf: report.tex und den Ordner Bilder/ hochladen.\n'
                  'Hauptdokument: report.tex. Compiler: XeLaTeX (pdfLaTeX ebenfalls möglich).\n'
                  'Keine Python-Ausführung, kein shell-escape und keine weiteren TeX-Teildateien nötig.\n'
                  'PDF-Abbildungen im Ordner Bilder sind Vektorgrafiken.\n')
    (destination/'README.txt').write_text(instructions)
    with zipfile.ZipFile(HERE/'Overleaf_FDM_BT.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination/'report.tex','report.tex')
        archive.write(destination/'README.txt','README.txt')
        for file in sorted(pictures.glob('*.pdf')):archive.write(file,'Bilder/'+file.name)
    with zipfile.ZipFile(HERE/'Bilder.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(pictures.glob('*.pdf')):archive.write(file,'Bilder/'+file.name)
    print(f'Overleaf: {len(needed)} Vektorabbildungen, ein eigenständiger TeX-Quelltext')


if __name__=='__main__':main()
