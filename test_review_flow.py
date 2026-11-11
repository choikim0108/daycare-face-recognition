import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import face_engine as engine
from photo_storage import copy_unique, move_review_photo, validate_directories


def face(embedding):
    return {'embedding': embedding, 'facial_area': {'w': 10, 'h': 10}}


class ReviewFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.input, self.refs, self.output = [root / p for p in ('input', 'refs', 'results')]
        for path in (self.input, self.refs / 'Alice', self.refs / 'Bob', self.output):
            path.mkdir(parents=True)
        (self.refs / 'Alice' / 'base.jpg').write_bytes(b'reference')
        (self.input / 'group.jpg').write_bytes(b'original')
        self.profiles = {'Alice': np.array([1., 0.]), 'Bob': np.array([0., 1.])}

    def process(self, faces=None, threshold=.32):
        with patch.object(engine, 'represent', return_value=faces or [face([1, 0]), face([0, 1])]):
            engine.process_photos(self.input, self.output, self.profiles,
                                  threshold=threshold, profile_dir=self.refs)

    def test_rerun_never_trains_unapproved_results(self):
        self.process()
        self.process()
        with patch.object(engine, 'get_largest_face_embedding', return_value=np.array([1., 0.])) as embed:
            engine.build_profiles(self.refs)
        self.assertEqual([Path(c.args[0]).name for c in embed.call_args_list], ['base.jpg'])
        self.assertEqual(len(list((self.output / 'Alice').glob('*.jpg'))), 2)
        self.assertEqual((self.input / 'group.jpg').read_bytes(), b'original')

    def test_only_explicit_approval_adds_reference(self):
        self.process([face([1, 0])])
        source = self.output / 'Alice' / 'group.jpg'
        with patch.object(engine, 'represent', return_value=[face([1, 0])]):
            added = engine.approve_reference_photo(source, self.output, self.refs, 'Alice')
        self.assertEqual(added.read_bytes(), b'original')
        self.assertTrue(source.exists())
        self.assertEqual((self.refs / 'Alice' / 'base.jpg').read_bytes(), b'reference')

    def test_group_and_no_face_cannot_be_approved(self):
        self.process()
        for faces in ([face([1, 0]), face([0, 1])], []):
            with patch.object(engine, 'represent', return_value=faces), self.assertRaises(ValueError):
                engine.approve_reference_photo(self.output / 'Alice' / 'group.jpg', self.output, self.refs, 'Alice')
        self.assertEqual(len(list((self.refs / 'Alice').iterdir())), 1)

    def test_review_move_preserves_group_other_child_and_original(self):
        self.process()
        target = move_review_photo(self.output / 'Alice' / 'group.jpg', self.output, 'Bob')
        self.assertEqual(target.name, 'group_1.jpg')
        for path in (target, self.output / 'Bob' / 'group.jpg', self.input / 'group.jpg'):
            self.assertEqual(path.read_bytes(), b'original')
        self.assertFalse((self.output / 'Alice' / 'group.jpg').exists())
        self.assertEqual(len(list((self.refs / 'Alice').iterdir())), 1)

    def test_repeated_collisions_do_not_overwrite(self):
        dest = self.output / 'Alice'
        results = [copy_unique(self.input / 'group.jpg', dest) for _ in range(5)]
        self.assertEqual(len(set(results)), 5)
        self.assertTrue(all(p.read_bytes() == b'original' for p in results))

    def test_same_child_move_is_noop(self):
        self.process()
        source = self.output / 'Alice' / 'group.jpg'
        self.assertEqual(move_review_photo(source, self.output, 'Alice'), source)
        self.assertTrue(source.exists())

    def test_all_equal_and_nested_folder_combinations_rejected(self):
        paths = [self.input, self.refs, self.output]
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                for nested in (False, True):
                    bad = paths.copy()
                    bad[b] = bad[a] / 'nested' if nested else bad[a]
                    with self.subTest(a=a, b=b, nested=nested), self.assertRaises(ValueError):
                        validate_directories(*bad)
        self.assertEqual((self.input / 'group.jpg').read_bytes(), b'original')

    def test_failed_copy_never_removes_review_source(self):
        self.process()
        source = self.output / 'Alice' / 'group.jpg'
        with patch('photo_storage.shutil.copyfileobj', side_effect=OSError('disk full')), self.assertRaises(OSError):
            move_review_photo(source, self.output, 'Bob')
        self.assertTrue(source.exists())
        self.assertFalse((self.output / 'Bob' / 'group_1.jpg').exists())

    def test_latest_twenty_cap_preserves_all_files(self):
        import os
        for i in range(25):
            path = self.refs / 'Alice' / f'{i}.jpg'
            path.write_bytes(b'placeholder')
            os.utime(path, (2000000000 + i, 2000000000 + i))
        with patch.object(engine, 'get_largest_face_embedding', return_value=np.array([1., 0.])) as embed:
            engine.build_profiles(self.refs)
        self.assertEqual(len(embed.call_args_list), 20)
        self.assertEqual({Path(c.args[0]).name for c in embed.call_args_list}, {f'{i}.jpg' for i in range(5, 25)})
        self.assertEqual(len(list((self.refs / 'Alice').iterdir())), 26)

    def test_unknown_and_detector_errors_preserve_original(self):
        with patch.object(engine, 'represent', side_effect=RuntimeError('no face')):
            engine.process_photos(self.input, self.output, self.profiles, profile_dir=self.refs)
        self.assertEqual((self.output / '미분류(Unknown)' / 'group.jpg').read_bytes(), b'original')
        self.assertTrue((self.input / 'group.jpg').exists())

    def test_threshold_is_respected(self):
        self.process([face([.8, .6])], threshold=.9)
        self.assertTrue((self.output / '미분류(Unknown)' / 'group.jpg').exists())

    def test_wrong_child_and_external_source_cannot_be_approved(self):
        self.process()
        for source, child in ((self.input / 'group.jpg', 'Alice'), (self.output / 'Alice' / 'group.jpg', 'Bob')):
            with self.assertRaises(ValueError):
                engine.approve_reference_photo(source, self.output, self.refs, child)

    def test_results_cannot_be_reused_as_input_or_profiles_later(self):
        self.process()
        other_output = self.output.parent / 'other-results'
        with self.assertRaises(ValueError):
            validate_directories(self.output, self.refs, other_output)
        with self.assertRaises(ValueError):
            validate_directories(self.input, self.output, other_output)
        with self.assertRaises(ValueError):
            engine.build_profiles(self.output / 'Alice')

    def test_storage_failure_is_reported_and_original_is_preserved(self):
        with patch.object(engine, 'represent', return_value=[face([1, 0])]), \
             patch('photo_storage.shutil.copyfileobj', side_effect=OSError('disk full')), \
             self.assertRaises(OSError):
            engine.process_photos(self.input, self.output, self.profiles, profile_dir=self.refs)
        self.assertEqual((self.input / 'group.jpg').read_bytes(), b'original')
        self.assertEqual((self.refs / 'Alice' / 'base.jpg').read_bytes(), b'reference')

    def test_approval_collision_preserves_existing_reference(self):
        self.process([face([1, 0])])
        existing = self.refs / 'Alice' / 'group.jpg'
        existing.write_bytes(b'existing reference')
        with patch.object(engine, 'represent', return_value=[face([1, 0])]):
            target = engine.approve_reference_photo(self.output / 'Alice' / 'group.jpg', self.output, self.refs, 'Alice')
        self.assertEqual(target.name, 'group_1.jpg')
        self.assertEqual(existing.read_bytes(), b'existing reference')

    def test_path_traversal_is_rejected_before_copy(self):
        with patch.object(engine, 'represent', return_value=[face([1, 0])]), self.assertRaises(ValueError):
            engine.process_photos(self.input, self.output, {'../refs': np.array([1., 0.])}, profile_dir=self.refs)
        self.assertEqual((self.refs / 'Alice' / 'base.jpg').read_bytes(), b'reference')


if __name__ == '__main__':
    unittest.main(verbosity=2)
