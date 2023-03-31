from nbconvert import PythonExporter
import nbformat
import os
import sys

wholedir = os.listdir("/Users/jchid/Downloads/Eyetracking Colab Notebooks/Model Files")

"Read file and determine the filetype"

for file in wholedir:
    while True:
        if file == jupyter:
            with open(file) as f:
                nb = nbformat.read(f, as_version=4)

            exporter = PythonExporter()
            source, meta = exporter.from_notebook_node(nb)
            pyfile = file.rstrip(" .ipynb") + ".py"

            with open(pyfile, 'w') as f:
                f.write(source)

            break

        elif file == json:
            "Convert file to jupyter notebook"

        elif file == python:
            break

        else:
            sys.exit("Unrecognized file type")

