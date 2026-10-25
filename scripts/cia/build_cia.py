#!/usr/bin/env python3
"""Builds the Super Mario Sunshine 3DS port's .cia (for the HOME Menu) from the player's own disc.

The icon (the HUD's Shine Sprite), the banner (the title logo) and the banner's sound (the title
theme's first seconds) are made from GMSE01.iso, so the release carries no Nintendo art or music.
The .3dsx next to this script gets the same icon, for the Homebrew Launcher.

    python3 build_cia.py [GMSE01.iso] [--out DIR]

Needs Python 3.8 or newer (no packages) and makerom and bannertool: tools/<windows|macos|linux>/
next to this script, else on the PATH.
"""
import argparse
import math
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gcdisc  # noqa: E402

# The HOME Menu's titles when there is no .3dsx to take them from (its SMDH, as the Makefile made it).
TITLE = 'Super Mario Sunshine'
DESCRIPTION = 'Native 3DS port (WIP) - requires your own GMSE01 disc image'
PUBLISHER = 'ikevik'
APP = 'sms3ds'


# --- images: rows of (r, g, b, a) ---------------------------------------------------------------

def resize(rows, w, h):
    """Area-averaged resize (premultiplied alpha)."""
    sh, sw = len(rows), len(rows[0])
    pm = [[(r * a, g * a, b * a, a) for r, g, b, a in row] for row in rows]

    def spans(n_src, n_dst):
        out = []
        for i in range(n_dst):
            lo, hi = i * n_src / n_dst, (i + 1) * n_src / n_dst
            taps = []
            for s in range(int(lo), min(n_src, int(math.ceil(hi)))):
                wgt = min(hi, s + 1) - max(lo, s)
                if wgt > 0:
                    taps.append((s, wgt / (hi - lo)))
            out.append(taps)
        return out

    xs, ys = spans(sw, w), spans(sh, h)
    tmp = [[tuple(sum(row[s][c] * wt for s, wt in taps) for c in range(4)) for taps in xs] for row in pm]
    out = []
    for taps in ys:
        line = []
        for x in range(w):
            r, g, b, a = (sum(tmp[s][x][c] * wt for s, wt in taps) for c in range(4))
            if a > 0:
                line.append((round(r / a), round(g / a), round(b / a), round(a)))
            else:
                line.append((0, 0, 0, 0))
        out.append(line)
    return out


def over(dst, src, x0, y0):
    """src composited over dst at (x0, y0)."""
    for y, row in enumerate(src):
        for x, (r, g, b, a) in enumerate(row):
            if a == 0:
                continue
            dr, dg, db, da = dst[y0 + y][x0 + x]
            k = a / 255
            dst[y0 + y][x0 + x] = (round(r * k + dr * (1 - k)), round(g * k + dg * (1 - k)),
                                   round(b * k + db * (1 - k)), max(a, da))


def mix(c0, c1, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))


def make_icon(disc):
    """48x48: the HUD's Shine Sprite on a sky glow (the HOME Menu icon has no transparency)."""
    _, shine = gcdisc.bti(gcdisc.rarc(disc.file('/data/game_6.szs'))['game_6/timg/shine_icon.bti'])
    icon = []
    for y in range(48):
        line = []
        for x in range(48):
            d = min(1.0, math.hypot(x - 23.5, y - 23.5) / 30)
            line.append(mix((255, 246, 200, 255), (40, 120, 220, 255), d ** 0.8))
        icon.append(line)
    w, h = len(shine[0]), len(shine)
    if w > 46 or h > 46:
        k = 46 / max(w, h)
        w, h = max(1, round(w * k)), max(1, round(h * k))
        shine = resize(shine, w, h)
    over(icon, shine, (48 - w) // 2, (48 - h) // 2)
    return icon


def make_banner(disc):
    """256x128: the title logo on a sky gradient."""
    _, logo = gcdisc.bti(gcdisc.rarc(disc.file('/data/option.szs'))['option/timg/title_logo.bti'])
    k = min(236 / len(logo[0]), 120 / len(logo))
    w, h = round(len(logo[0]) * k), round(len(logo) * k)
    logo = resize(logo, w, h)
    banner = [[mix((70, 150, 235, 255), (190, 230, 250, 255), y / 127)] * 256 for y in range(128)]
    banner = [list(row) for row in banner]
    over(banner, logo, (256 - w) // 2, (128 - h) // 2)
    return banner


def make_jingle(disc, seconds=2.95, fade=0.35):
    """The title theme's first seconds, faded out (a banner's sound: 2 channels, 3 s at most)."""
    data = disc.file('/AudioRes/Streams/title.afc')
    rate = struct.unpack('>H', data[8:10])[0]
    n = int(rate * seconds)
    rate, left, right = gcdisc.afc_stream(data, n)
    fin, fout = int(rate * 0.005), int(rate * fade)
    for i in range(len(left)):
        g = min(1.0, i / fin, (len(left) - i) / fout)
        left[i], right[i] = int(left[i] * g), int(right[i] * g)
    return rate, left, right


# --- tools ----------------------------------------------------------------------------------------

def platform_dir():
    if os.name == 'nt':
        return 'windows'
    return 'macos' if sys.platform == 'darwin' else 'linux'


def find_tool(name, tools_dir):
    exe = name + ('.exe' if os.name == 'nt' else '')
    p = os.path.join(tools_dir, platform_dir(), exe)
    if os.path.isfile(p):
        if os.name != 'nt':
            os.chmod(p, os.stat(p).st_mode | 0o111)
            if sys.platform == 'darwin':  # unzipped downloads are quarantined: Gatekeeper would block them
                subprocess.run(['xattr', '-d', 'com.apple.quarantine', p], capture_output=True)
        return p
    p = shutil.which(name)
    if p:
        return p
    sys.exit('error: %s not found (put it in %s or on the PATH)' % (name, os.path.join(tools_dir, platform_dir())))


def run(cmd):
    print('  $ ' + ' '.join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit('error: %s failed (exit %d)' % (os.path.basename(cmd[0]), r.returncode))


def smdh_range(data, path):
    """Where a .3dsx keeps its SMDH: (offset, size)."""
    if data[:4] != b'3DSX' or struct.unpack('<H', data[4:6])[0] <= 0x20:
        sys.exit('error: %s has no SMDH' % path)
    return struct.unpack('<II', data[0x20:0x28])


def smdh_titles(path):
    """A .3dsx's English short title, long title and publisher."""
    data = open(path, 'rb').read()
    off, _ = smdh_range(data, path)
    entry = data[off + 8 + 0x200:off + 8 + 0x400]  # the second of 16 entries: English

    def text(a, n):
        return entry[a:a + n].decode('utf-16-le', 'replace').split('\0')[0]

    return text(0, 0x80), text(0x80, 0x100), text(0x180, 0x80)


def patch_3dsx_icon(src, dst, smdh_path):
    """A copy of the .3dsx with its SMDH (icon and titles) replaced."""
    data = bytearray(open(src, 'rb').read())
    smdh = open(smdh_path, 'rb').read()
    off, size = smdh_range(data, src)
    if size != len(smdh):
        sys.exit('error: unexpected SMDH size in %s' % src)
    data[off:off + size] = smdh
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, 'wb').write(data)


def ask_iso():
    found = [f for f in os.listdir(HERE) if f.lower().endswith('.iso')]
    if len(found) == 1:
        return os.path.join(HERE, found[0])
    if os.name == 'nt':
        try:
            import tkinter
            from tkinter import filedialog
            tkinter.Tk().withdraw()
            p = filedialog.askopenfilename(title='Your Super Mario Sunshine disc image (GMSE01)',
                                           filetypes=[('Disc image', '*.iso'), ('All files', '*.*')])
            if p:
                return p
        except Exception:
            pass
    p = input('Path to your Super Mario Sunshine disc image (GMSE01.iso; you can drag it here): ')
    return p.strip().strip('"').strip("'").replace('\\ ', ' ')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('iso', nargs='?', help='your Super Mario Sunshine (USA, GMSE01) disc image')
    ap.add_argument('--out', default=os.path.join(HERE, 'output'), help='output folder')
    ap.add_argument('--elf', default=os.path.join(HERE, APP + '.elf'))
    ap.add_argument('--3dsx', dest='tdsx', default=os.path.join(HERE, APP + '.3dsx'))
    ap.add_argument('--rsf', default=os.path.join(HERE, APP + '.rsf'))
    ap.add_argument('--tools', default=os.path.join(HERE, 'tools'))
    ap.add_argument('--version', default=None, help='title version, as 0.1.0 (default: version.txt)')
    args = ap.parse_args()

    iso = args.iso or ask_iso()
    if not os.path.isfile(iso):
        sys.exit('error: no disc image at %s' % iso)
    if not os.path.isfile(args.elf):
        sys.exit('error: %s is missing' % args.elf)
    version = args.version
    vfile = os.path.join(HERE, 'version.txt')
    if version is None and os.path.isfile(vfile):
        version = open(vfile).read().strip()
    major, minor, micro = ([int(''.join(ch for ch in p if ch.isdigit()) or 0)
                            for p in (version or '0.0.0').split('-')[0].split('.')] + [0, 0, 0])[:3]

    makerom = find_tool('makerom', args.tools)
    bannertool = find_tool('bannertool', args.tools)

    disc = gcdisc.Disc(iso)
    if disc.game_id != 'GMSE01':
        sys.exit('error: %s is %s, not Super Mario Sunshine (USA, GMSE01)' % (iso, disc.game_id or '?'))

    work = os.path.join(args.out, 'art')
    os.makedirs(work, exist_ok=True)
    print('Making the icon, banner and banner sound from your disc...')
    gcdisc.write_png(os.path.join(work, 'icon.png'), make_icon(disc))
    gcdisc.write_png(os.path.join(work, 'banner.png'), make_banner(disc))
    gcdisc.write_wav(os.path.join(work, 'jingle.wav'), *make_jingle(disc))

    title, description, publisher = TITLE, DESCRIPTION, PUBLISHER
    if os.path.isfile(args.tdsx):
        title, description, publisher = smdh_titles(args.tdsx)
    icn, bnr = os.path.join(work, 'icon.icn'), os.path.join(work, 'banner.bnr')
    run([bannertool, 'makesmdh', '-s', title, '-l', description, '-p', publisher,
         '-i', os.path.join(work, 'icon.png'), '-f', 'visible,allow3d,recordusage,new3ds', '-o', icn])
    run([bannertool, 'makebanner', '-i', os.path.join(work, 'banner.png'),
         '-a', os.path.join(work, 'jingle.wav'), '-o', bnr])

    cia = os.path.join(args.out, APP + '.cia')
    print('Building the CIA...')
    run([makerom, '-f', 'cia', '-o', cia, '-rsf', args.rsf, '-target', 't', '-elf', args.elf,
         '-icon', icn, '-banner', bnr, '-exefslogo',
         '-major', str(major), '-minor', str(minor), '-micro', str(micro)])

    sd = os.path.join(args.out, 'sd-card')
    if os.path.isfile(args.tdsx):
        patch_3dsx_icon(args.tdsx, os.path.join(sd, '3ds', APP, APP + '.3dsx'), icn)
    settings = os.path.join(HERE, 'settings.txt')
    if os.path.isfile(settings):
        os.makedirs(os.path.join(sd, 'sms'), exist_ok=True)
        shutil.copyfile(settings, os.path.join(sd, 'sms', 'settings.txt'))

    print()
    print('Done:')
    print('  ' + cia)
    if os.path.isdir(sd):
        print('  ' + sd + '  (for the SD card: the .3dsx with the same icon, and the settings)')
    print()
    print('Next: install %s.cia with FBI, copy the sd-card folder\'s contents to your SD card, and put' % APP)
    print('your disc image there as /sms/GMSE01.iso (the game reads it from there; the CIA holds none of it).')
    print('Please don\'t share the built CIA: its icon, banner and sound come from your disc.')


if __name__ == '__main__':
    main()
