import json

with open('langgraph-basics/2-simple-chatbot.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell.get('outputs'):
        print(f'=== Cell {i} ===')
        for output in cell['outputs']:
            if output.get('output_type') == 'error':
                print(f'  ERROR: {output.get("ename", "Unknown")}: {output.get("evalue", "")}')
                print('  Traceback:')
                for line in output.get('traceback', []):
                    print('    ' + line)
            elif output.get('output_type') == 'execute_result':
                print(f'  Result: {str(output.get("data", {}))[:600]}')
            elif output.get('output_type') == 'stream':
                print(f'  Stream: {str(output.get("text", ""))[:300]}')