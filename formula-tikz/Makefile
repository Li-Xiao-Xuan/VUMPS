default: fig

%: %.tex
	pdflatex -interaction nonstopmode -halt-on-error -file-line-error $<
	-@latexmk -c $<
	-@echo 'compiled! ✨'

%.tar: %.tex %.pdf
	tar -cvf a.$@ $^ Makefile

clean:
	-@latexmk -c
	-@trash *.bib *.bbl
	-@echo 'clean up! 🍰'
