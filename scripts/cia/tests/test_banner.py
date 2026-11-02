"""Asset-free failure checks plus optional validation of an actual generated model."""
import hashlib
import http.client
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import wave

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import banner_verify as verify
import bootstrap_banner as bootstrap
import relief_banner as relief


class BannerTests(unittest.TestCase):
    def test_relief_keeps_the_front_projection_at_different_depths(self):
        profile = {'camera_distance': 40., 'center_y': 1.}
        config = dict(columns=1, rows=1, depths=[[2., 8.], [12., 4.]], coverage=['1'],
                      center_x=3., width=10., height=6.)
        vertices, uv, faces = relief.surface(profile, config)
        for actual, expected in zip(vertices, [(-2., 4.), (8., 4.), (-2., -2.), (8., -2.)]):
            x, y, depth = actual
            scale = (40 - depth) / 40
            self.assertAlmostEqual(x / scale, expected[0])
            self.assertAlmostEqual(1 + (y - 1) / scale, expected[1])
        self.assertEqual(len(faces), 2)
        self.assertEqual(uv, [[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

    def test_invisible_grid_cell_has_no_faces(self):
        config = dict(columns=1, rows=1, depths=[[1., 1.], [1., 1.]], coverage=['0'],
                      center_x=0., width=2., height=2.)
        self.assertEqual(relief.surface({'camera_distance': 40., 'center_y': 0.}, config)[2], [])

    def test_invalid_profile_depth_is_rejected(self):
        config = dict(columns=1, rows=1, depths=[[1., float('nan')], [1., 1.]], coverage=['1'],
                      center_x=0., width=2., height=2.)
        with self.assertRaisesRegex(ValueError, 'depth'):
            relief.surface({'camera_distance': 40., 'center_y': 0.}, config)

    def test_lz11_checks_limit_truncation_and_backreferences(self):
        for data in [b'\x11\x01\x00\x08', b'\x11\x03\0\0\0a', b'\x11\x03\0\0\x80\x20\0']:
            with self.subTest(data=data), self.assertRaises(ValueError):
                verify.lz11(data)
        self.assertEqual(verify.lz11(b'\x11\x03\0\0\0abc'), b'abc')

    def test_custom_audio_is_not_trimmed_or_changed(self):
        pcm = b'\x01\x02\x03\x04' * 17
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'voice.wav'
            with wave.open(str(path), 'wb') as output:
                output.setparams((2, 2, 32000, 17, 'NONE', 'not compressed'))
                output.writeframes(pcm)
            self.assertEqual(verify.validate_wave(path), (32000, 17, pcm))

    def test_overlong_and_mono_audio_fail_instead_of_being_fixed_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            for channels, count in [(1, 10), (2, 96001)]:
                path = Path(tmp) / 'voice.wav'
                with wave.open(str(path), 'wb') as output:
                    output.setparams((channels, 2, 32000, count, 'NONE', 'not compressed'))
                    output.writeframes(bytes(channels * 2 * count))
                with self.subTest(channels=channels), self.assertRaises(ValueError):
                    verify.validate_wave(path)

    def test_non_wav_audio_has_a_plain_format_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'voice.mp3'
            path.write_bytes(b'not a RIFF wave file')
            with self.assertRaisesRegex(ValueError, 'PCM WAV'):
                verify.validate_wave(path)

    def test_audio_alignment_moves_the_offset_without_touching_the_resource(self):
        data = bytearray(0x98)
        data[:4] = b'CBMD'
        struct.pack_into('<I', data, 0x84, 0x98)
        audio = b'CWAV-unmodified-resource'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'banner.bnr'
            path.write_bytes(data + audio)
            verify.align_audio(path)
            result = path.read_bytes()
            offset = verify.u32(result, 0x84)
            self.assertEqual(offset % 32, 0)
            self.assertEqual(result[offset:], audio)

    def test_download_retries_an_incomplete_response_and_verifies_the_retry(self):
        payload = b'the complete pinned source'
        digest = hashlib.sha256(payload).hexdigest()
        with mock.patch.object(bootstrap, 'SHA256', digest), mock.patch.object(
                bootstrap.urllib.request, 'urlopen',
                side_effect=[http.client.IncompleteRead(b'partial'), io.BytesIO(payload)]) as fetch:
            self.assertEqual(bootstrap.download_converter(), payload)
            self.assertEqual(fetch.call_count, 2)

    def test_download_never_accepts_the_wrong_archive(self):
        with mock.patch.object(bootstrap.urllib.request, 'urlopen',
                               side_effect=lambda *a, **kw: io.BytesIO(b'wrong source')):
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                bootstrap.download_converter()

    def test_archive_member_cannot_escape_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bootstrap.safe_destination(Path(tmp), '../escape.py')

    def test_older_python_rejects_relief_before_creating_a_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'unused'
            with mock.patch.object(bootstrap.sys, 'version_info', (3, 10)):
                with self.assertRaisesRegex(RuntimeError, 'Python 3.11'):
                    bootstrap.ensure_tools(cache)
            self.assertFalse(cache.exists())


@unittest.skipUnless(os.environ.get('SMS_TEST_BANNER_DIR'), 'set SMS_TEST_BANNER_DIR for generated-model checks')
class GeneratedModelTests(unittest.TestCase):
    def setUp(self):
        self.data = (Path(os.environ['SMS_TEST_BANNER_DIR']) / 'banner.cgfx').read_bytes()

    def test_generated_model_matches_the_safe_profile(self):
        report = verify.validate_cgfx(self.data)
        self.assertEqual(report['triangles'], 6372)
        self.assertLess(report['cgfx_bytes'], 256 * 1024)

    def test_an_out_of_bounds_triangle_is_rejected(self):
        data = self.data
        model = verify.dictionary(data, 0x1c)[0][1]
        shape = verify.pointers(data, model + 0xc4)[0]
        primitive_set = verify.pointers(data, shape + 0x2c)[0]
        primitive = verify.pointers(data, primitive_set + 12)[0]
        stream = verify.pointers(data, primitive)[0]
        payload = verify.relative(data, stream + 12)
        corrupted = bytearray(data)
        struct.pack_into('<H', corrupted, payload, 65535)
        with self.assertRaisesRegex(ValueError, 'index exceeds'):
            verify.validate_cgfx(corrupted)


if __name__ == '__main__':
    unittest.main()
