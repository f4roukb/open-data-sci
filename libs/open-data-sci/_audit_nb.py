import json
out = open('_audit_nb_out.txt', 'w', encoding='utf-8')
for f in ['examples/notebooks/030_notebook_anthropic.ipynb','examples/notebooks/031_notebook_openai_compatible_server.ipynb','examples/notebooks/032_notebook_bedrock.ipynb']:
    nb = json.load(open(f, encoding='utf-8'))
    out.write('='*20 + f + '='*20 + '\n')
    for cell in nb['cells']:
        src = ''.join(cell['source'])
        out.write('--- ' + cell['cell_type'] + ' ---\n')
        out.write(src)
        out.write('\n\n')
out.close()
