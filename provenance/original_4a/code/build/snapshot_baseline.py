"""Freeze constructed baseline inputs, reference row and extracted source routines.

One-time source acquisition. No regression is run. This stage reads the declared
broader-folder source artifacts but imports none of their modules.
"""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

APP = Path(__file__).resolve().parents[2]
ROOT = APP.parents[1]
SHARED = Path('/Users/f1azmm00/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg')
SOURCE = ROOT/'batch_logs/macro_paper_application/source51_replay_20160914'
DEST = SHARED/'4_Macro_version/4a_PaperApplication/data/intermediate/baseline_roll5_gap5'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    for p, digest in manifest['source_sha256'].items():
        if '/site-packages/' not in p:
            assert sha(p)==digest, f'Producer changed: {p}'
    DEST.mkdir(parents=True,exist_ok=False)
    copied = {}
    for name,digest in manifest['input_cache_sha256'].items():
        src = Path(manifest['input_cache'])/name
        assert sha(src)==digest
        shutil.copy2(src,DEST/name)
        assert sha(DEST/name)==digest
        copied[name]=digest
    reference_path = APP/'results/coefficients/source51_grid_2016.csv'
    ref = pd.read_csv(reference_path)
    ref = ref[ref.exposure.eq('roll5_gap5') & ref.fe.eq('county+year')]
    assert len(ref)==1 and ref.n_obs.iloc[0]==113146 and ref.sample_last.iloc[0]==2016
    fixture = APP/'tests/fixtures/baseline_roll5_gap5_reference.csv'
    assert not fixture.exists()
    ref.to_csv(fixture,index=False)
    declarations = [
        (ROOT/'4_Macro_version/step_1/code/py/51_county_multiplier_grid.py',['lam_weights','wins']),
        (ROOT/'4_Macro_version/step_2/code/py/22_winsor_construction_error.py',['demean']),
        (ROOT/'4_Macro_version/step_2/code/py/28_table_d1.py',['drop_singletons','is_nested','ssc_k','crv1','fit2sls'])]
    functions = []
    provenance = []
    for path,names in declarations:
        source = path.read_text()
        tree = ast.parse(source)
        by_name = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in names:
            node = by_name[name]
            provenance.append({'source':str(path.relative_to(ROOT)), 'function':name,
                               'source_file_sha256':sha(path),
                               'source_function_sha256':hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()})
            if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
            rendered = ast.unparse(node).replace('w22.demean(', 'demean(')
            functions.append(rendered)
    target = APP/'code/lib/baseline_math.py'
    assert not target.exists()
    target.write_text('"""Source-extracted arithmetic for the locked county baseline.\n\nExtraction provenance: config/baseline_sources.json.\n"""\nimport numpy as np\nimport pandas as pd\n\n\n'+'\n\n\n'.join(functions)+'\n')
    provenance_file = APP/'config/baseline_sources.json'
    data = {'boundary':'frozen_constructed_inputs_not_raw_to_result',
            'snapshot_directory':'data/intermediate/baseline_roll5_gap5',
            'inputs':copied, 'reference_sha256':sha(fixture),
            'reference_source_sha256':sha(reference_path),
            'math_sha256':sha(target),'extracted_functions':provenance,
            'county_screen':'producer51 whole-history screen through 2018; frozen 3058 counties'}
    provenance_file.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'snapshot':str(DEST),'reference':str(fixture),'status':'PASS'},indent=2))


if __name__=='__main__':
    main()
