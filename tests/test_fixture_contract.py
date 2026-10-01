"""Dependency-free safety and CLI checks; not numerical integration evidence."""
import ast
import pathlib
import subprocess
import sys
import unittest

HARNESS = pathlib.Path(__file__).with_name('fixture_integration.py')


class FixtureContractTests(unittest.TestCase):
    def test_source_lf_and_syntax(self):
        raw = HARNESS.read_bytes()
        self.assertNotIn(b'\r', raw)
        self.assertTrue(raw.endswith(b'\n'))
        self.assertFalse(raw.endswith(b'\n\n'))
        ast.parse(raw)

    def cli(self, *arguments):
        return subprocess.run([sys.executable, '-B', str(HARNESS), *arguments], capture_output=True, text=True, timeout=10)

    def test_help_does_not_need_numerical_dependencies(self):
        result = self.cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--seed {0,1}', result.stdout)

    def test_missing_seed_is_rejected(self):
        self.assertEqual(self.cli().returncode, 2)

    def test_out_of_contract_seed_is_rejected(self):
        self.assertEqual(self.cli('--seed', '2').returncode, 2)

    def test_unknown_option_is_rejected(self):
        self.assertEqual(self.cli('--seed', '0', '--train').returncode, 2)

    def test_heavy_imports_are_deferred(self):
        tree = ast.parse(HARNESS.read_text())
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.assertFalse({'torch', 'numpy'} & {name.name for name in statement.names})
            if isinstance(statement, ast.ImportFrom):
                self.assertNotIn(statement.module, ('torch', 'numpy', 'st_saca'))

    def test_no_training_entry_or_optimizer_construction(self):
        tree = ast.parse(HARNESS.read_text())
        forbidden = {'train_saca', 'evaluate_policy', 'SAC', 'Adam', 'AdamW', 'SGD', 'backward', 'supervised_loss'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                self.assertNotIn(name, forbidden)


if __name__ == '__main__':
    unittest.main()
