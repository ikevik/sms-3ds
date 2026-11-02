"""Bounds and profile checks for generated HOME banner resources (stdlib only)."""
import hashlib
import math
import struct
import wave


def u32(data, offset):
    if offset < 0 or offset + 4 > len(data):
        raise ValueError('Truncated banner field')
    return struct.unpack_from('<I', data, offset)[0]


def relative(data, offset):
    value = u32(data, offset)
    if value == 0:
        return None
    if value >= 0x80000000:
        value -= 0x100000000
    target = offset + value
    if not 0 <= target < len(data):
        raise ValueError('Banner pointer outside its resource')
    return target


def dictionary(data, offset):
    count = u32(data, offset)
    if not count:
        return []
    table = relative(data, offset + 4)
    if table is None or data[table:table + 4] != b'DICT' or u32(data, table + 8) != count:
        raise ValueError('Invalid banner dictionary')
    result = []
    for i in range(count):
        name = relative(data, table + 36 + 16 * i)
        end = data.find(b'\0', name) if name is not None else -1
        if end == -1:
            raise ValueError('Unterminated banner resource name')
        result.append((data[name:end].decode('utf-8'), relative(data, table + 40 + 16 * i)))
    return result


def pointers(data, offset):
    count = u32(data, offset)
    table = relative(data, offset + 4)
    if count and table is None:
        raise ValueError('Missing banner pointer array')
    return [relative(data, table + 4 * i) for i in range(count)]


def lz11(data):
    if len(data) < 4 or data[0] != 0x11:
        raise ValueError('Expected LZ11 banner model')
    size = int.from_bytes(data[1:4], 'little')
    offset = 4
    if not size:
        size, offset = u32(data, 4), 8
    if not 0 < size <= 0x80000:
        raise ValueError('Decompressed CGFX exceeds the 512 KiB format limit')
    result = bytearray()
    try:
        while len(result) < size:
            flags = data[offset]
            offset += 1
            for bit in range(7, -1, -1):
                if len(result) >= size:
                    break
                a = data[offset]
                offset += 1
                if not flags & (1 << bit):
                    result.append(a)
                    continue
                high = a >> 4
                if high == 0:
                    b, c = data[offset:offset + 2]
                    offset += 2
                    length, distance = ((a & 15) << 4 | b >> 4) + 0x11, ((b & 15) << 8 | c) + 1
                elif high == 1:
                    b, c, d = data[offset:offset + 3]
                    offset += 3
                    length = ((a & 15) << 12 | b << 4 | c >> 4) + 0x111
                    distance = ((c & 15) << 8 | d) + 1
                else:
                    b = data[offset]
                    offset += 1
                    length, distance = high + 1, ((a & 15) << 8 | b) + 1
                if distance > len(result):
                    raise ValueError('Invalid LZ11 back-reference')
                for _ in range(min(length, size - len(result))):
                    result.append(result[-distance])
    except (IndexError, ValueError) as exc:
        raise ValueError('Truncated or invalid LZ11 banner model') from exc
    return bytes(result)


def validate_cgfx(data):
    if data[:4] != b'CGFX' or u32(data, 12) != len(data):
        raise ValueError('Invalid CGFX header/length')
    if len(data) >= 256 * 1024:
        raise ValueError('Relief exceeds its conservative 256 KiB model budget')
    models = dictionary(data, 0x1c)
    if len(models) != 1 or models[0][0] != 'COMMON':
        raise ValueError('Expected one COMMON banner model')
    model = models[0][1]
    skeleton = relative(data, model + 0xe0)
    for _, bone in dictionary(data, skeleton + 0x18):
        if u32(data, bone + 0xd4) != 4:
            raise ValueError('Relief bone is not fixed-facing')
    shapes = pointers(data, model + 0xc4)
    triangles = 0
    for shape in shapes:
        vertices = None
        for attribute in pointers(data, shape + 0x38):
            kind, usage = u32(data, attribute), u32(data, attribute + 4)
            if usage in (7, 8):
                raise ValueError('Soft-skin attributes are unsafe for this HOME profile')
            if kind == 0x80000000:
                if usage != 1 or u32(data, attribute + 24) != 3:
                    raise ValueError('Unexpected constant banner attribute')
                values = relative(data, attribute + 28)
                if values is None or data[values:values + 12] != struct.pack('<fff', 0, 0, 1):
                    raise ValueError('Invalid fixed normal')
                continue
            if kind != 0x40000001 or u32(data, attribute + 36) != 0x1402:
                raise ValueError('Expected compact signed-16 banner attributes')
            components = u32(data, attribute + 40)
            scale = struct.unpack('<f', data[attribute + 44:attribute + 48])[0]
            length, payload = u32(data, attribute + 20), relative(data, attribute + 24)
            if components not in (2, 3) or not math.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid compact vertex scale')
            if payload is None or payload + length > len(data) or length % (2 * components):
                raise ValueError('Invalid vertex buffer length')
            if usage == 0:
                vertices = length // 6
        if vertices is None:
            raise ValueError('Missing banner positions')
        for primitive_set in pointers(data, shape + 0x2c):
            if u32(data, primitive_set + 8) != 0 or u32(data, primitive_set) != 1:
                raise ValueError('Expected a rigid single-bone primitive')
            for primitive in pointers(data, primitive_set + 12):
                for stream in pointers(data, primitive):
                    size, payload = u32(data, stream + 8), relative(data, stream + 12)
                    if u32(data, stream) != 0x1403 or size % 6 or payload is None or payload + size > len(data):
                        raise ValueError('Invalid triangle index stream')
                    indices = struct.unpack('<' + 'H' * (size // 2), data[payload:payload + size])
                    if max(indices, default=0) >= vertices:
                        raise ValueError('Triangle index exceeds its vertex buffer')
                    triangles += len(indices) // 3
    animations = dictionary(data, 0x64)
    if len(animations) != 1 or animations[0][0] != 'COMMON':
        raise ValueError('Missing relief animation')
    animation = animations[0][1]
    if struct.unpack_from('<f', data, animation + 20)[0] != 480:
        raise ValueError('Unexpected relief loop duration')
    members = dictionary(data, animation + 24)
    if {name for name, _ in members} != {'art_far_surface', 'art_volume_surface'}:
        raise ValueError('Unexpected animated relief nodes')
    keys = []
    for _, member in members:
        if u32(data, member) != 0x2bf0000 or u32(data, member + 16) != 5:
            raise ValueError('Expected ordinary Y-only Transform animation')
        curve = relative(data, member + 52)
        segment = relative(data, curve + 20)
        if u32(data, segment + 12) != 6:
            raise ValueError('Unexpected number of relief motion keys')
        raw = data[segment + 20:segment + 92]
        if len(raw) != 72:
            raise ValueError('Truncated relief motion')
        values = struct.unpack('<' + 'fff' * 6, raw)
        if list(values[::3]) != [0, 80, 160, 240, 400, 480]:
            raise ValueError('Unexpected relief motion timing or unsafe high bob')
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Invalid relief motion values')
        keys.append(raw)
    if keys[0] != keys[1]:
        raise ValueError('Mario and water must move together')
    if triangles != 6372:
        raise ValueError('Relief topology changed unexpectedly')
    return dict(cgfx_bytes=len(data), triangles=triangles, fixed_facing=True, animation_frames=480)


def validate_wave(path):
    try:
        with wave.open(str(path), 'rb') as audio:
            if audio.getnchannels() != 2 or audio.getsampwidth() != 2 or audio.getcomptype() != 'NONE':
                raise ValueError('Banner audio must be an uncompressed stereo 16-bit PCM WAV')
            rate, frames = audio.getframerate(), audio.getnframes()
            if not rate or not 0 < frames <= 3 * rate:
                raise ValueError('Banner audio must be nonempty and no longer than three seconds')
            pcm = audio.readframes(frames)
            if len(pcm) != frames * 4:
                raise ValueError('Truncated banner WAV')
    except (wave.Error, EOFError) as exc:
        raise ValueError('Cannot read PCM WAV: %s' % exc) from exc
    return rate, frames, pcm


def align_audio(path):
    data = path.read_bytes()
    offset = u32(data, 0x84)
    padding = (-offset) % 32
    if padding:
        result = bytearray(data[:offset] + bytes(padding) + data[offset:])
        struct.pack_into('<I', result, 0x84, offset + padding)
        path.write_bytes(result)


def validate_banner(path, wav_path, cgfx_path=None):
    data = path.read_bytes()
    if data[:4] != b'CBMD':
        raise ValueError('Invalid banner container')
    start, audio_offset = u32(data, 8), u32(data, 0x84)
    if not 0x88 <= start < audio_offset < len(data):
        raise ValueError('Invalid banner resource offsets')
    model = lz11(data[start:audio_offset])
    report = dict(banner_bytes=len(data), cgfx_bytes=len(model))
    if cgfx_path is not None:
        if len(data) >= 512 * 1024:
            raise ValueError('Banner exceeds the conservative 512 KiB relief package budget; '
                             'use a smaller PCM WAV or --banner flat')
        if model != cgfx_path.read_bytes():
            raise ValueError('Packed banner model differs from generated CGFX')
        report.update(validate_cgfx(model))
        if audio_offset % 32:
            raise ValueError('Relief audio must be 32-byte aligned')
    audio = data[audio_offset:]
    if audio[:4] != b'CWAV' or u32(audio, 12) != len(audio):
        raise ValueError('Invalid embedded CWAV length')
    info, payload = u32(audio, 24), u32(audio, 36)
    if audio[info:info + 4] != b'INFO' or audio[payload:payload + 4] != b'DATA':
        raise ValueError('Invalid CWAV sections')
    rate, frames, pcm = validate_wave(wav_path)
    if audio[info + 8:info + 10] != b'\x01\x00' or u32(audio, info + 12) != rate:
        raise ValueError('Unexpected CWAV encoding, loop mode or sample rate')
    if u32(audio, info + 20) != frames or u32(audio, info + 28) != 2:
        raise ValueError('CWAV frame/channel count mismatch')
    for channel in range(2):
        channel_info = info + 28 + u32(audio, info + 36 + channel * 8)
        offset = payload + 8 + u32(audio, channel_info + 4)
        expected = b''.join(pcm[i:i + 2] for i in range(channel * 2, len(pcm), 4))
        if audio[offset:offset + len(expected)] != expected:
            raise ValueError('Embedded CWAV does not preserve the input PCM')
    report.update(audio_rate=rate, audio_frames=frames, pcm_preserved=True,
                  cwav_sha256=hashlib.sha256(audio).hexdigest())
    return report
