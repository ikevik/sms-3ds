"""Reading a GameCube disc image without any packages: its file system, Yaz0-compressed RARC
archives, BTI textures (to RGBA) and JAudio AFC streams (to PCM), plus PNG and WAV writers.

Used by build_cia.py to make the HOME Menu icon, banner and banner sound from the player's own
Super Mario Sunshine disc (GMSE01), so that no Nintendo art or music ships with the port.
"""
import struct
import zlib


class Disc:
    """A GameCube disc image's files by path (as "/data/title.szs")."""

    def __init__(self, path):
        self.f = open(path, 'rb')
        head = self.read(0, 0x440)
        self.game_id = head[0:6].decode('latin1')
        self.files = {}
        fst_off, fst_size = struct.unpack('>II', head[0x424:0x42C])
        fst = self.read(fst_off, fst_size)
        n = struct.unpack('>I', fst[8:12])[0]
        strtab = n * 12

        def name(i):
            o = strtab + (struct.unpack('>I', fst[i * 12:i * 12 + 4])[0] & 0xFFFFFF)
            return fst[o:fst.index(b'\0', o)].decode('latin1')

        stack = [(n, '')]
        for i in range(1, n):
            while i >= stack[-1][0]:
                stack.pop()
            path = stack[-1][1] + '/' + name(i)
            a, b = struct.unpack('>II', fst[i * 12 + 4:i * 12 + 12])
            if fst[i * 12]:
                stack.append((b, path))
            else:
                self.files[path.lower()] = (a, b)

    def read(self, off, size):
        self.f.seek(off)
        data = self.f.read(size)
        if len(data) != size:
            raise IOError('the disc image is shorter than its file table says')
        return data

    def file(self, path):
        off, size = self.files[path.lower()]
        return self.read(off, size)


def yaz0(data):
    """Yaz0-decompressed data (data itself when it is not compressed)."""
    if data[:4] != b'Yaz0':
        return data
    size = struct.unpack('>I', data[4:8])[0]
    out = bytearray(size)
    src, dst = 16, 0
    while dst < size:
        code = data[src]
        src += 1
        for bit in range(7, -1, -1):
            if dst >= size:
                break
            if code & (1 << bit):
                out[dst] = data[src]
                src += 1
                dst += 1
            else:
                b1, b2 = data[src], data[src + 1]
                src += 2
                back = dst - (((b1 & 0xF) << 8) | b2) - 1
                n = b1 >> 4
                if n == 0:
                    n = data[src] + 0x12
                    src += 1
                else:
                    n += 2
                for k in range(n):
                    out[dst + k] = out[back + k]
                dst += n
    return bytes(out)


def rarc(data):
    """A RARC archive's files: {path: bytes}, paths as "title/timg/logo.bti" (lower case)."""
    data = yaz0(data)
    if data[:4] != b'RARC':
        raise ValueError('not a RARC archive')
    data_off = struct.unpack('>I', data[0x0C:0x10])[0] + 0x20
    info = 0x20
    num_nodes, node_off, num_ent, ent_off, str_size, str_off = struct.unpack(
        '>IIIIII', data[info:info + 0x18])
    node_off += info
    ent_off += info
    str_off += info

    def string(o):
        o += str_off
        return data[o:data.index(b'\0', o)].decode('latin1')

    files = {}

    def walk(node, prefix):
        o = node_off + node * 0x10
        _, name_off, _, count, first = struct.unpack('>4sIHHI', data[o:o + 0x10])
        name = string(name_off)
        path = prefix + name + '/'
        for e in range(first, first + count):
            q = ent_off + e * 0x14
            _, _, flags, noff, a, b = struct.unpack('>HHHHII', data[q:q + 0x10])
            ename = string(noff & 0xFFFF)
            if ename in ('.', '..'):
                continue
            if (flags >> 8) & 0x02:
                walk(a, path)
            else:
                files[(path + ename).lower()] = data[data_off + a:data_off + a + b]

    walk(0, '')
    return files


def _rgb5a3(v):
    if v & 0x8000:
        r, g, b = (v >> 10) & 31, (v >> 5) & 31, v & 31
        return (r << 3 | r >> 2, g << 3 | g >> 2, b << 3 | b >> 2, 255)
    a, r, g, b = (v >> 12) & 7, (v >> 8) & 15, (v >> 4) & 15, v & 15
    return (r * 17, g * 17, b * 17, a << 5 | a << 2 | a >> 1)


def _rgb565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return (r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2, 255)


def _ia8(v):
    i, a = v & 0xFF, v >> 8
    return (i, i, i, a)


def gx_texture(fmt, w, h, data, palette=None):
    """A GX texture in its tiled format as rows of (r, g, b, a). palette: a list of rgba for C4/C8."""
    out = [[(0, 0, 0, 0)] * w for _ in range(h)]
    bw, bh, bpp = {0: (8, 8, 4), 1: (8, 4, 8), 2: (8, 4, 8), 3: (4, 4, 16), 4: (4, 4, 16),
                   5: (4, 4, 16), 6: (4, 4, 32), 8: (8, 8, 4), 9: (8, 4, 8), 14: (8, 8, 4)}[fmt]
    pos = 0
    for by in range(0, h, bh):
        for bx in range(0, w, bw):
            if fmt == 14:  # CMPR: 2x2 DXT1 sub-blocks
                for sb in range(4):
                    c0, c1, bits = struct.unpack('>HHI', data[pos:pos + 8])
                    pos += 8
                    p0, p1 = _rgb565(c0), _rgb565(c1)
                    if c0 > c1:
                        p2 = tuple((2 * x + y) // 3 for x, y in zip(p0, p1))
                        p3 = tuple((x + 2 * y) // 3 for x, y in zip(p0, p1))
                    else:
                        p2 = tuple((x + y) // 2 for x, y in zip(p0, p1))
                        p3 = (0, 0, 0, 0)
                    pal = (p0, p1, p2, p3)
                    for k in range(16):
                        x = bx + (sb & 1) * 4 + (k & 3)
                        y = by + (sb >> 1) * 4 + (k >> 2)
                        if x < w and y < h:
                            out[y][x] = pal[(bits >> (30 - 2 * k)) & 3]
                continue
            if fmt == 6:  # RGBA8: AR then GB planes
                for k in range(16):
                    x, y = bx + (k & 3), by + (k >> 2)
                    a, r = data[pos + 2 * k], data[pos + 2 * k + 1]
                    g, b = data[pos + 32 + 2 * k], data[pos + 33 + 2 * k]
                    if x < w and y < h:
                        out[y][x] = (r, g, b, a)
                pos += 64
                continue
            for k in range(bw * bh):
                x, y = bx + k % bw, by + k // bw
                if bpp == 4:
                    v = data[pos + k // 2]
                    v = v >> 4 if k % 2 == 0 else v & 15
                elif bpp == 8:
                    v = data[pos + k]
                else:
                    v = struct.unpack('>H', data[pos + 2 * k:pos + 2 * k + 2])[0]
                if fmt == 0:
                    px = (v * 17,) * 3 + (255,)
                elif fmt == 1:
                    px = (v,) * 3 + (255,)
                elif fmt == 2:  # IA4: alpha in the high nibble
                    px = ((v & 15) * 17,) * 3 + ((v >> 4) * 17,)
                elif fmt == 3:
                    px = _ia8(v)
                elif fmt == 4:
                    px = _rgb565(v)
                elif fmt == 5:
                    px = _rgb5a3(v)
                else:  # C4, C8
                    px = palette[v] if palette and v < len(palette) else (0, 0, 0, 0)
                if x < w and y < h:
                    out[y][x] = px
            pos += bw * bh * bpp // 8
    return out


def bti(data):
    """A BTI texture: (format, rows of rgba)."""
    fmt, _, w, h = struct.unpack('>BBHH', data[0:6])
    pal_fmt = data[9]
    pal_count = struct.unpack('>H', data[0x0A:0x0C])[0]
    pal_off = struct.unpack('>I', data[0x0C:0x10])[0]
    img_off = struct.unpack('>I', data[0x1C:0x20])[0]
    palette = None
    if fmt in (8, 9):
        conv = (_ia8, _rgb565, _rgb5a3)[pal_fmt]
        palette = [conv(struct.unpack('>H', data[pal_off + 2 * i:pal_off + 2 * i + 2])[0])
                   for i in range(pal_count)]
    return fmt, gx_texture(fmt, w, h, data[img_off:], palette)


_AFC_COEF = (0, 0, 0x800, 0, 0, 0x800, 0x400, 0x400, 0x1000, -0x800, 0xE00, -0x600, 0xC00, -0x400,
             0x1200, -0xA00, 0x1068, -0x8C8, 0x12C0, -0x8FC, 0x1400, -0xC00, 0x800, -0x800,
             0x400, -0x400, -0x400, 0x400, -0x400, 0, -0x800, 0)


def afc_stream(data, max_samples=None):
    """A JAudio stream (.afc, as JAInter::StreamLib reads it): (sample rate, left, right).

    Format 4 is ADPCM: 18-byte blocks of a 9-byte left frame and a 9-byte right frame, 16 samples
    each (a header byte, high nibble the scale, low nibble the predictor), as __DecodeADPCM."""
    size, nsamples, rate, fmt = struct.unpack('>IIHH', data[0:12])
    body = data[0x20:0x20 + size]
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    left, right = [], []
    if fmt == 2:  # PCM16, the channels in 0x2800-byte halves of each 0x5000-byte block
        for blk in range(0, len(body), 0x5000):
            half = (len(body) - blk) // 2 if len(body) - blk < 0x5000 else 0x2800
            n = half // 2
            left += struct.unpack('>%dh' % n, body[blk:blk + 2 * n])
            right += struct.unpack('>%dh' % n, body[blk + half:blk + half + 2 * n])
            if len(left) >= nsamples:
                break
        return rate, left[:nsamples], right[:nsamples]
    if fmt != 4:
        raise ValueError('unknown stream format %d' % fmt)
    hist = [[0, 0], [0, 0]]
    chans = (left, right)
    for blk in range(0, len(body) - 17, 18):
        for ch in range(2):
            p = blk + ch * 9
            hdr = body[p]
            c1, c2 = _AFC_COEF[(hdr & 15) * 2], _AFC_COEF[(hdr & 15) * 2 + 1]
            scale = hdr >> 4
            h1, h2 = hist[ch]
            out = chans[ch]
            for k in range(16):
                b = body[p + 1 + (k >> 1)]
                r = (b >> 4) if k % 2 == 0 else b & 15
                r = r - 16 if r >= 8 else r
                s = (r << scale) + ((c1 * h1 + c2 * h2) >> 11)
                s = ((s + 0x8000) & 0xFFFF) - 0x8000  # the decoder stores to s16
                out.append(s)
                h2, h1 = h1, s
            hist[ch] = [h1, h2]
        if len(left) >= nsamples:
            break
    return rate, left[:nsamples], right[:nsamples]


def write_png(path, rows):
    """rows of (r, g, b, a) as an RGBA PNG."""
    h, w = len(rows), len(rows[0])
    raw = b''.join(b'\0' + bytes(c for px in row for c in px) for row in rows)

    def chunk(tag, body):
        return struct.pack('>I', len(body)) + tag + body + struct.pack('>I', zlib.crc32(tag + body))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 9)))
        f.write(chunk(b'IEND', b''))


def write_wav(path, rate, left, right):
    """16-bit stereo PCM."""
    frames = b''.join(struct.pack('<hh', l, r) for l, r in zip(left, right))
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', 36 + len(frames)) + b'WAVE')
        f.write(b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 2, rate, rate * 4, 4, 16))
        f.write(b'data' + struct.pack('<I', len(frames)) + frames)
