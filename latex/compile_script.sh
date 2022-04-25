#!/bin/sh

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
rm *.aux *.log *.brf *.blg *.bbl
