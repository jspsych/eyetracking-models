from nbconvert import PythonExporter
import nbformat
import os
import sys

wholedir = os.listdir("/Users/jchid/Downloads/Eyetracking Colab Notebooks/Models")

"Read file and determine the filetype"

for file in wholedir:
        
        if file.endswith(".ipynb"):
            with open(file) as f:
                nb = nbformat.read(f, as_version=4)

            exporter = PythonExporter()
            source, meta = exporter.from_notebook_node(nb)
            pyfile = file.rstrip(" .ipynb") + ".py"

            with open("/Users/jchid/Downloads/Eyetracking Colab Notebooks/Models(Python)/" + pyfile, 'w') as f:
                f.write(source)


        elif file.endswith(".py"):
            pass

        else:
            sys.exit(f"{file} has unrecognized file type")

