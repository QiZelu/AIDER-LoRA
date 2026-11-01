"""Verify completed-run comparison labels, checkpoint hashes and overwrite refusal."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.audit import file_digest
from engine.trainer import compare_control,train
from utils.protocol import CONTROL,make_config


class ReportTests(unittest.TestCase):
    def test_comparison_uses_matched_labels_and_reports_candidate_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders={name:Path(tmp)/name for name in ('joint','staged')}
            for name,folder in folders.items():
                folder.mkdir();(folder/'config.yaml').write_text('unit');(folder/'sources.json').write_text('{}')
                for kind in ('best','last'):
                    report=json.loads((CONTROL/f'{kind}_evaluation.json').read_text())
                    checkpoint=folder/f'{kind}.pt';checkpoint.write_bytes(b'unit-checkpoint')
                    report['checkpoint_sha256']=file_digest(checkpoint)
                    (folder/f'{kind}_evaluation.json').write_text(json.dumps(report))
            with patch('engine.trainer.CONTROL',folders['joint']),patch('engine.trainer.validate_reference'):
                compare_control(folders['staged'])
                result=json.loads((folders['staged']/'comparison_joint_vs_two_stage.json').read_text())
                self.assertEqual(len(result['rows']),40)
                self.assertEqual(result['provenance']['joint']['best_epochs'],[1,10])
                self.assertEqual(result['provenance']['two_stage']['best_epochs'],[6,10])
                self.assertTrue(all(row['difference']==0 for row in result['rows']))
                path=folders['staged']/'best_evaluation.json'
                report=json.loads(path.read_text());report['samples'][0]['target']+=.1
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'labels differ'):compare_control(folders['staged'])

    def test_train_refuses_existing_claim_before_cuda_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.training_claim').write_text('existing')
            config=make_config(root=root)
            with patch('engine.trainer.load_prepared'),patch('engine.trainer.initialize_process_memory') as cuda:
                with self.assertRaises(FileExistsError):train(config)
                cuda.assert_not_called()


if __name__=='__main__':unittest.main()
