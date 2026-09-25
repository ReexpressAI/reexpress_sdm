# Copyright Reexpress AI, Inc. All rights reserved.
"""Public CLI spelling, help, and safe defaults for new and continued models."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reexpress_sdm import (
    SDMModel, TorchTrainer, TrainingConfig, build_artifact, load_artifact,
    train_iterations, write_artifact, write_dataset_bundle,
)
from reexpress_sdm.cli import build_parser, main
from helpers import make_artifact


class CLIDefaultTests(unittest.TestCase):
    @staticmethod
    def _parsers(parser):
        yield parser
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    yield from CLIDefaultTests._parsers(child)

    def test_help_shows_defaults_and_every_option_has_an_explanation(self):
        for parser in self._parsers(build_parser()):
            with self.subTest(command=parser.prog):
                self.assertFalse(parser.allow_abbrev)
                self.assertEqual(parser.formatter_class, argparse.ArgumentDefaultsHelpFormatter)
                for action in parser._actions:
                    if action.option_strings:
                        self.assertTrue(action.help)
                        self.assertTrue(all('-' not in option[2:] for option in action.option_strings if option.startswith('--')))
                help_text = parser.format_help()
                if parser.prog == 'sdm train':
                    self.assertIn('(default: 1000)', help_text)
                    self.assertIn('(default: 1e-05)', help_text)
                    self.assertIn('embedding_v1', help_text)
                    self.assertIn('precomputed', help_text)
                    self.assertIn('Class0, Class1', help_text)
                elif parser.prog == 'sdm score':
                    self.assertIn('(default: 25)', help_text)

    def test_selection_defaults_to_centroid_and_lower_remains_explicit(self):
        for parser in self._parsers(build_parser()):
            if any('--estimator' in action.option_strings for action in parser._actions):
                with self.subTest(command=parser.prog):
                    self.assertEqual(parser.get_default('estimator'), 'centroid')
                    self.assertIn('(default: centroid)', ' '.join(parser.format_help().split()))
        args = build_parser().parse_args(['score', '--model', 'model.sdmkitmodel', '--input', 'eval.jsonl',
                                         '--estimator', 'lower'])
        self.assertEqual(args.estimator, 'lower')

    def test_removed_aliases_and_abbreviations_are_rejected(self):
        parser = build_parser()
        common = ['train', '--training', 'train', '--calibration', 'cal',
                  '--output', 'out', '--number_of_classes', '2']
        removed = [
            '--exemplar_vector_dimension', '--input_training_set_file',
            '--input_calibration_set_file', '--epoch', '--seed_value',
            '--main_device', '--maxQAvailableFromIndexer', '--number-of-classes',
            '--exemplar-dimension', '--batch-size', '--learning-rate',
            '--initial-model', '--cross-entropy-epochs', '--max-neighbors',
            '--representation-fingerprint', '--representation-provider',
            '--representation-model', '--report-output', '--number-of-random-shuffles',
            '--matching-query-batch-size', '--matching-support-tile-size', '--alpha-resolution',
            '--class-names', '--exemplar_dim',
            '--model_dir', '--class_size',
        ]
        for flag in removed:
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(common + [flag, '1'])
        for flag in ['--shuffle-training-and-calibration', '--do-not-shuffle-data']:
            with self.subTest(flag=flag), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(common + [flag])
        self.assertEqual(parser.parse_args(common).epochs, 20)

        for command in ['calibrate', 'serve']:
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([command])

        for command in ['score', 'evaluate']:
            arguments = [command, '--model', 'model.sdmkitmodel', '--input', 'eval.jsonl']
            self.assertFalse(hasattr(parser.parse_args(arguments), 'minimum_alpha'))
            command_parser = next(child for child in self._parsers(parser) if child.prog == f'sdm {command}')
            self.assertNotIn('--minimum_alpha', command_parser.format_help())
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments + ['--minimum_alpha', '0.95'])

    @staticmethod
    def _rows(prefix):
        return [
            {'id': f'{prefix}-{index}', 'label': index % 2, 'embedding': vector}
            for index, vector in enumerate([[1., 0.], [0., 1.], [2., 0.], [0., 2.]])
        ]

    def _train_cli(self, directory, *, fingerprint=None, names=None, initial=None, extra=(), expected=0):
        training, calibration = directory / 'train.sdmdataset', directory / 'cal.sdmdataset'
        representation = {'fingerprint': fingerprint} if fingerprint is not None else None
        for path, prefix in [(training, 'train'), (calibration, 'cal')]:
            write_dataset_bundle(path, self._rows(prefix), class_names=names,
                                 representation=representation, overwrite=True)
        output = directory / 'out.sdmkitmodel'
        args = ['train', '--training', str(training), '--calibration', str(calibration),
                '--output', str(output), '--number_of_classes', '2', '--epochs', '1',
                '--exemplar_dimension', '2', '--batch_size', '2', '--max_neighbors', '4',
                '--alpha_resolution', '.1', '--device', 'cpu', '--do_not_shuffle_data', '--overwrite']
        if initial is not None:
            source = directory / 'source.sdmkitmodel'
            write_artifact(source, initial, overwrite=True)
            args += ['--initial_model', str(source)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(args + list(extra))
        self.assertEqual(status, expected, stderr.getvalue())
        return load_artifact(output) if expected == 0 else stderr.getvalue()

    def test_cli_new_training_defaults_and_dataset_declarations(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            new = self._train_cli(directory)
            self.assertEqual(new.manifest['configuration']['classNames'], ['Class0', 'Class1'])
            self.assertEqual(new.manifest['representation']['fingerprint'], 'embedding_v1')
            self.assertEqual(new.manifest['representation']['provider'], 'precomputed')
            self.assertIsNone(new.manifest['representation']['model'])
            declared = self._train_cli(directory, fingerprint='declared-v2', names=['Before', 'After'])
            self.assertEqual(declared.manifest['representation']['fingerprint'], 'declared-v2')
            self.assertEqual(declared.manifest['configuration']['classNames'], ['Before', 'After'])
            explicit = self._train_cli(directory, extra=['--representation_fingerprint', 'explicit-v3',
                '--representation_provider', 'my-provider', '--representation_model', 'my-encoder',
                '--class_names', 'False,True'])
            self.assertEqual(explicit.manifest['representation']['fingerprint'], 'explicit-v3')
            self.assertEqual(explicit.manifest['representation']['provider'], 'my-provider')
            self.assertEqual(explicit.manifest['representation']['model'], 'my-encoder')
            self.assertEqual(explicit.manifest['configuration']['classNames'], ['False', 'True'])
            self._train_cli(directory, fingerprint='declared-v2',
                extra=['--representation_fingerprint', 'other'], expected=2)

    def test_cli_continuation_inherits_saved_identity_and_names(self):
        initial = make_artifact()
        initial.manifest['representation']['provider'] = 'saved-provider'
        initial.manifest['representation']['model'] = 'saved-model'
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            continued = self._train_cli(directory, initial=initial)
            self.assertEqual(continued.manifest['representation'], initial.manifest['representation'])
            self.assertEqual(continued.manifest['configuration']['classNames'], ['zero', 'one'])
            self._train_cli(directory, initial=initial, fingerprint='different', expected=2)
            self._train_cli(directory, initial=initial, names=['Class0', 'Class1'], expected=2)

    def test_conflicting_split_fingerprints_fail_before_training(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for split, fingerprint in [('train', 'first'), ('cal', 'second')]:
                write_dataset_bundle(root / f'{split}.sdmdataset', self._rows(split),
                                     representation={'fingerprint': fingerprint})
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                status = main(['train', '--training', str(root / 'train.sdmdataset'),
                    '--calibration', str(root / 'cal.sdmdataset'), '--output', str(root / 'out.sdmkitmodel'),
                    '--number_of_classes', '2'])
            self.assertEqual(status, 2)
            self.assertIn('fingerprints differ', stderr.getvalue())
            self.assertFalse((root / 'out.sdmkitmodel').exists())

    def test_sdk_training_defaults_and_continuation_match_cli(self):
        vectors = np.asarray([row['embedding'] for row in self._rows('sdk')], dtype=np.float32)
        labels = [0, 1, 0, 1]
        config = TrainingConfig(2, exemplar_dimension=2, epochs=1, batch_size=2,
                                max_neighbors=4, alpha_resolution=.1)
        fresh = TorchTrainer(config, device='cpu').fit(vectors, labels, vectors, labels).artifact
        self.assertEqual(fresh.manifest['configuration']['classNames'], ['Class0', 'Class1'])
        self.assertEqual(fresh.manifest['representation']['fingerprint'], 'embedding_v1')
        self.assertEqual(fresh.manifest['representation']['provider'], 'precomputed')
        self.assertIsNone(fresh.manifest['representation']['model'])
        initial = make_artifact()
        initial.manifest['representation']['provider'] = 'custom-provider'
        initial.manifest['representation']['model'] = 'custom-model'
        for iterative in (False, True):
            kwargs = dict(initial_artifact=initial)
            if iterative:
                result = train_iterations(config, vectors, labels, vectors, labels,
                    backend_options={'device': 'cpu'}, shuffle_training_and_calibration=False, **kwargs)
            else:
                result = TorchTrainer(config, device='cpu').fit(vectors, labels, vectors, labels, **kwargs)
            self.assertEqual(result.artifact.manifest['representation'], initial.manifest['representation'])
            self.assertEqual(result.artifact.manifest['configuration']['classNames'], ['zero', 'one'])

    def test_direct_artifact_builder_defaults_and_scoring_retains_model_identity(self):
        source = make_artifact()
        artifact = build_artifact(weights=source.weights, support_vectors=source.support_vectors,
            support_records=source.support_records, distance_cdfs=[[0., 1.], [0., 1.]],
            rescaled_similarity_cdfs=[[0., 1.], [0., 1.]], regions=[], embedding_dimension=2,
            exemplar_dimension=2, number_of_classes=2)
        self.assertEqual(artifact.manifest['configuration']['classNames'], ['Class0', 'Class1'])
        self.assertEqual(artifact.manifest['representation']['fingerprint'], 'embedding_v1')
        model = SDMModel(source, device='cpu')
        scores = model.score([[1., 0.]])
        self.assertEqual(len(scores), 1)
        self.assertEqual(model.artifact.manifest['representation']['fingerprint'], 'fixture-v1')


if __name__ == '__main__':
    unittest.main()
