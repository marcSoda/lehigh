from jinja2 import Environment, FileSystemLoader
import copy
import sys, os
import relational
import importlib

def main():

    
    #print(model)

    outfile = sys.argv[1]
    projDir = sys.argv[2]
    moduleName = sys.argv[3]

    relmodel = importlib.import_module(moduleName)

    model = relmodel.getModel() 

    
    file_loader = FileSystemLoader(projDir + "/tools/templates")
    env = Environment(loader=file_loader)
    template = env.get_template("relation.h.j2")

    output = template.render(model=model)

    #print(output)

    try:
        os.mkdir(os.path.dirname(outfile))
    except FileExistsError:
        print("PYTHON: Directory already exists")

    with open(outfile, 'w') as f:
        f.write(output)

if __name__ == "__main__":
    main()
