"""Build the compact, fixed-facing HOME relief from the bundled profile and disc logo."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcdisc


def checked_assets(directory):
    directory = Path(directory)
    for line in (directory / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if Path(name).name != name:
            raise ValueError('Invalid banner asset manifest')
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Banner asset checksum mismatch: ' + name)
    profile = json.loads((directory / 'relief.json').read_text())
    if profile['version'] != 1 or profile['texture_resolution'] != 256:
        raise ValueError('Unsupported relief profile')
    if len((directory / 'mario.etc1a4').read_bytes()) != 256 * 256:
        raise ValueError('Invalid native Mario texture length')
    return profile


def surface(profile, config):
    """Bake perspective-compensated vertices; preserve the authored coverage."""
    columns, rows = config['columns'], config['rows']
    depths = config['depths']
    if len(depths) != rows + 1 or any(len(row) != columns + 1 for row in depths):
        raise ValueError('Invalid depth grid')
    if len(config['coverage']) != rows or any(len(row) != columns for row in config['coverage']):
        raise ValueError('Invalid relief coverage')
    distance, cy = profile['camera_distance'], profile['center_y']
    vertices, uv = [], []
    for y in range(rows + 1):
        for x in range(columns + 1):
            u, v, depth = x / columns, y / rows, depths[y][x]
            if not math.isfinite(depth) or not 0 <= depth < distance:
                raise ValueError('Invalid relief depth')
            scale = (distance - depth) / distance
            vertices.append([(config['center_x'] + (u - .5) * config['width']) * scale,
                             cy + (.5 - v) * config['height'] * scale, depth])
            # glTF's V points down the image; the native CGFX texture is bottom-up.
            uv.append([u, v])
    faces = []
    for y in range(rows):
        for x in range(columns):
            if config['coverage'][y][x] == '0':
                continue
            if config['coverage'][y][x] != '1':
                raise ValueError('Invalid coverage cell')
            a = y * (columns + 1) + x
            c = a + columns + 1
            faces.extend([(a, c, a + 1), (a + 1, c, c + 1)])
    faces.sort(key=lambda face: sum(vertices[i][2] for i in face))
    return vertices, uv, faces


def edge_colours(image):
    """Bleed invisible RGB for linear sampling, without changing alpha."""
    width, height = image.size
    pixels = list(image.get_flattened_data())
    nearest = []
    for y in range(height):
        valid = [x for x in range(width) if pixels[y * width + x][3] > 8]
        nearest.append([min(valid, key=lambda p: (abs(p - x), p)) if valid else None
                        for x in range(width)])
    result = list(pixels)
    for x in range(width):
        candidates = [(y, nearest[y][x]) for y in range(height) if nearest[y][x] is not None]
        if not candidates:
            continue
        for y in range(height):
            i = y * width + x
            if pixels[i][3] > 8:
                continue
            py, px = min(candidates, key=lambda p: ((p[0] - y) ** 2 + (p[1] - x) ** 2, p[1], p[0]))
            result[i] = pixels[py * width + px][:3] + (pixels[i][3],)
    image.putdata(result)
    return image


def scene(profile, logo_path, assets, out):
    """Write an ordinary rigid glTF; conversion below supplies CGFX-only fields."""
    binary = bytearray()
    views, accessors, meshes = [], [], []

    def accessor(values, code, components, component_type, target):
        while len(binary) % 4:
            binary.append(0)
        start = len(binary)
        flat = [n for value in values for n in value] if components > 1 else values
        binary.extend(struct.pack('<' + code * len(flat), *flat))
        index = len(views)
        views.append(dict(buffer=0, byteOffset=start, byteLength=len(binary) - start, target=target))
        result = len(accessors)
        accessors.append(dict(bufferView=index, componentType=component_type, count=len(values),
                              type={1: 'SCALAR', 2: 'VEC2', 3: 'VEC3'}[components]))
        return result

    def mesh(name, vertices, uv, faces, material):
        used = sorted({i for face in faces for i in face})
        remap = {value: i for i, value in enumerate(used)}
        attributes = {'POSITION': accessor([vertices[i] for i in used], 'f', 3, 5126, 34962),
                      'NORMAL': accessor([[0., 0., 1.]] * len(used), 'f', 3, 5126, 34962),
                      'TEXCOORD_0': accessor([uv[i] for i in used], 'f', 2, 5126, 34962)}
        indices = accessor([remap[i] for face in faces for i in face], 'H', 1, 5123, 34963)
        meshes.append(dict(name=name, primitives=[dict(attributes=attributes, indices=indices, material=material)]))

    vertices, uv, faces = surface(profile, profile['mario'])
    threshold = profile['mario']['split_depth']
    far = [f for f in faces if sum(vertices[i][2] for i in f) / 3 < threshold]
    near = [f for f in faces if sum(vertices[i][2] for i in f) / 3 >= threshold]
    mesh('art_far_surface', vertices, uv, far, 0)
    mesh('art_volume_surface', vertices, uv, near, 0)
    vertices, uv, faces = surface(profile, profile['logo'])
    mesh('LogoRelief', vertices, uv, faces, 1)
    materials = []
    for name, texture in [('Hero_Artwork_Unlit', 0), ('SunshineLogo_Unlit', 1)]:
        materials.append(dict(name=name, alphaMode='BLEND', doubleSided=False,
                              pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1],
                                  baseColorTexture=dict(index=texture), metallicFactor=0, roughnessFactor=1)))
    document = dict(asset=dict(version='2.0', generator='sms3ds relief banner'), scene=0,
                    scenes=[dict(nodes=[3])],
                    nodes=[dict(name='art_far_surface', mesh=0), dict(name='art_volume_surface', mesh=1),
                           dict(name='name', mesh=2), dict(name='COMMON', children=[0, 1, 2])],
                    meshes=meshes, materials=materials,
                    images=[dict(name='MarioArtwork', uri=Path(os.path.relpath(assets / 'mario.png', out)).as_posix()),
                            dict(name='SunshineLogo', uri=logo_path.name)],
                    samplers=[dict(magFilter=9729, minFilter=9987, wrapS=33071, wrapT=33071)],
                    textures=[dict(source=0, sampler=0), dict(source=1, sampler=0)],
                    buffers=[dict(uri='relief.bin', byteLength=len(binary))],
                    bufferViews=views, accessors=accessors)
    (out / 'relief.bin').write_bytes(binary)
    (out / 'relief.gltf').write_text(json.dumps(document) + '\n')
    return out / 'relief.gltf'


def convert(gltf_path, assets, profile, pycgfx):
    sys.path.insert(0, str(pycgfx))
    import gltflib
    import main as converter
    from cgfx.shared import StandardObject, List, ListData
    from cgfx.txob import ImageTexture, TextureFormat
    from cgfx.sobj import BillboardMode, BoneFlag, SkeletonFlag
    from cgfx.primitives import VertexAttributeUsage as Usage, DataType, VertexParamAttribute
    from cgfx.mtob import MTOBFlag, FragmentLightingFlags, AlphaTestFunction, DepthFlag, BlendFunction
    from cgfx.canm import (CANM, CANMBoneTransform, FloatAnimationCurve, FloatSegment,
                          UnifiedHermite96Key, InterpolationType, QuantizationType, RepeatMethod)

    class MeshVisibility(StandardObject):
        struct = struct.Struct('iI')

        def __init__(self, name):
            self.name = name

        def values(self):
            return self.name, 1

    class FloatData(ListData):
        def refresh_struct(self):
            self.struct = struct.Struct('f' * len(self.contents))

    original_texture = converter.gltf_get_texture

    def texture(cgfx, gltf, image_id, normal=False):
        image = gltf.model.images[image_id]
        if image.name != 'MarioArtwork':
            return original_texture(cgfx, gltf, image_id, normal)
        if normal:
            raise ValueError('Mario artwork is not a normal map')
        cached = cgfx.data.textures[image.name]
        if cached is not None:
            return cached
        result = ImageTexture()
        result.name = image.name
        result.width = result.height = result.pixel_based_image.width = result.pixel_based_image.height = 256
        result.hw_format = TextureFormat.ETC1A4
        result.mipmap_level_count = 1
        result.pixel_based_image.data = (assets / 'mario.etc1a4').read_bytes()
        cgfx.data.textures.add(image.name, result)
        return result

    converter.gltf_get_texture = texture
    try:
        cgfx = converter.convert_gltf(gltflib.GLTF.load(str(gltf_path), load_file_resources=True))
    finally:
        converter.gltf_get_texture = original_texture
    model = cgfx.data.models['COMMON']
    bone_names = list(model.skeleton.bones)
    owners = []
    for mesh in model.meshes.data.contents:
        shape = model.shapes.data.contents[mesh.shape_index]
        primitives = shape.primitive_sets.data.contents
        if len(primitives) != 1 or primitives[0].skinning_mode != 0:
            raise ValueError('HOME banner must contain only rigid geometry')
        related = primitives[0].related_bones.data.contents
        if len(related) != 1:
            raise ValueError('Ambiguous rigid mesh binding')
        owner = bone_names[related[0]]
        mesh.mesh_node_name = owner
        if owner not in owners:
            owners.append(owner)
            model.mesh_nodes.add(owner, MeshVisibility(owner))
        mesh.mesh_node_visibility_index = model.mesh_nodes.get_index(owner)
        mesh.priority = {'name': 0, 'art_far_surface': 1, 'art_volume_surface': 2}[owner]

    for name in model.materials:
        material = model.materials[name]
        material.flags = MTOBFlag(0)
        material.rasterization.cull_mode = material.rasterization.command.param = 0
        fragment = material.fragment_shader
        fragment.fragment_lighting.flags = FragmentLightingFlags(0)
        fragment.fragment_lighting_table.distribution_0_sampler = None
        for i, stage in enumerate(fragment.texture_combiners):
            stage.src_rgb = stage.src_alpha = 0x333 if i == 0 else 0xFFF
            stage.combine_rgb = stage.combine_alpha = 0
        fragment.alpha_test.enabled = True
        fragment.alpha_test.function = AlphaTestFunction.Greater
        fragment.alpha_test.cutoff = 0
        material.fragment_operations.depth_operation.flags = DepthFlag.TestEnabled
        blend = material.fragment_operations.blend_operation
        blend.src_color, blend.dst_color = BlendFunction.SrcAlpha, BlendFunction.InvSrcAlpha
        blend.src_alpha, blend.dst_alpha = BlendFunction.One, BlendFunction.InvSrcAlpha

    animated = ('art_far_surface', 'art_volume_surface')
    for name in model.skeleton.bones:
        bone = model.skeleton.bones[name]
        bone.billboard_mode = BillboardMode.ScreenViewpoint
        bone.flags = BoneFlag(0x19f if name not in owners else 0x1df)
        if name in animated:
            bone.flags &= ~(BoneFlag.IsIdentity | BoneFlag.IsTranslateZero)
    model.skeleton.flags = SkeletonFlag.IsTranslateAnimationEnabled
    animation = CANM()
    animation.name, animation.target_animation_group_name = 'COMMON', 'SkeletalAnimation'
    animation.looping, animation.frame_size = True, float(profile['motion']['frames'])
    for name in animated:
        member = CANMBoneTransform()
        member.bone_path = name
        curve = FloatAnimationCurve()
        curve.start_frame, curve.end_frame = profile['motion']['curve_start'], profile['motion']['curve_end']
        curve.pre_repeat_method = curve.post_repeat_method = RepeatMethod.Repeat
        curve.flags = 4
        segment = FloatSegment()
        segment.start_frame, segment.end_frame = curve.start_frame, curve.end_frame
        segment.interpolation, segment.quantization = InterpolationType.CubicSpline, QuantizationType.UnifiedHermite96
        for frame, value, slope in profile['motion']['keys']:
            key = UnifiedHermite96Key()
            key.frame, key.value, key.in_out_slope = frame, value, slope
            segment.keys.append(key)
        curve.segments.append(segment)
        member.pos_y = curve
        animation.member_animations_data.add(name, member)
    cgfx.data.skeletal_animations.add('COMMON', animation)

    # Compact typed attributes retain the topology and subpixel precision.
    # The texture-only material ignores normals, so no per-vertex normals need
    # allocating or uploading. The high-resolution depth profile stays intact.
    for shape in model.shapes.data.contents:
        for index, attribute in enumerate(shape.vertex_attributes.data.contents):
            if attribute.usage == Usage.Normal:
                fixed = VertexParamAttribute()
                fixed.type, fixed.usage = -0x80000000, Usage.Normal
                fixed.components_count, fixed.scale = 3, 1.
                fixed.attributes = List()
                fixed.attributes.data = FloatData([0., 0., 1.])
                shape.vertex_attributes.data.contents[index] = fixed
                continue
            if attribute.usage not in (Usage.Position, Usage.TextureCoordinate0):
                raise ValueError('Unexpected vertex attribute in relief mesh')
            denominator = profile['position_scale'] if attribute.usage == Usage.Position else profile['uv_scale']
            raw = attribute.vertex_stream_data
            values = struct.unpack('<' + 'f' * (len(raw) // 4), raw)
            integers = [round(value * denominator) for value in values]
            if not all(-32767 <= value <= 32767 for value in integers):
                raise ValueError('Relief vertex exceeds its compact attribute range')
            attribute.vertex_stream_data = struct.pack('<' + 'h' * len(integers), *integers)
            attribute.format_type, attribute.scale = DataType.Short, 1 / denominator
    model.skeleton.bones.dict.regenerate()
    model.mesh_nodes.dict.regenerate()
    result = converter.write(cgfx)
    if len(result) >= 256 * 1024:
        raise ValueError('Relief exceeds the tested 256 KiB model budget')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iso', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pycgfx', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=HERE / 'banner_assets')
    args = parser.parse_args()
    try:
        from PIL import Image
        profile = checked_assets(args.assets)
        with Image.open(args.assets / 'mario.png') as image:
            if list(image.size) != profile['mario']['source_size']:
                raise ValueError('Mario artwork does not match the authored profile')
        disc = gcdisc.Disc(str(args.iso))
        try:
            if disc.game_id != 'GMSE01':
                raise ValueError('Expected a GMSE01 disc image')
            files = gcdisc.rarc(disc.file('/data/option.szs'))
            _, pixels = gcdisc.bti(files['option/timg/title_logo.bti'])
        finally:
            disc.f.close()
        if [len(pixels[0]), len(pixels)] != profile['logo']['source_size']:
            raise ValueError('Disc logo does not match the authored relief profile')
        args.out.mkdir(parents=True, exist_ok=True)
        logo = Image.new('RGBA', (len(pixels[0]), len(pixels)))
        logo.putdata([pixel for row in pixels for pixel in row])
        logo = edge_colours(logo.resize((256, 128), Image.Resampling.LANCZOS))
        logo_path = args.out / 'logo.png'
        logo.save(logo_path)
        gltf = scene(profile, logo_path, args.assets, args.out)
        result = convert(gltf, args.assets, profile, args.pycgfx)
        (args.out / 'banner.cgfx').write_bytes(result)
        report = dict(cgfx_bytes=len(result), model_budget_bytes=256 * 1024,
                      cgfx_sha256=hashlib.sha256(result).hexdigest(), billboard_mode=4,
                      animation_frames=profile['motion']['frames'], rigid_only=True,
                      mario_texture=[256, 256], mario_source='banner_assets/README.md',
                      logo_source='GMSE01:/data/option.szs:option/timg/title_logo.bti')
        (args.out / 'relief-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Built compact relief banner (%d bytes)' % len(result))
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, 'Relief banner error: %s\n' % exc)


if __name__ == '__main__':
    main()
