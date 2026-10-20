# Native J3D drawing (design note)

Status: written 2026-10-19 against f3f58f7; stage 0 (attribution) under way since 2026-10-20. Since then
master has gained GX state resets as blocks (46d5f77, 25b2f63) and replay of cached state-only lists
as register words (d9e46c6), so some of section 1's GX-thread numbers are a few ms lower now.

Unless a number is marked as an estimate, it was measured on a New 3DS in Delfino Plaza: the profile run
of build Oct 19 2026 15:17:47 (60 frames standing still; see appendix A for how the log was read).

## Summary

- **What.** Draw J3D packets from state compiled once per material (a `PicaState` made by the backend's
  own TEV translation) and geometry compiled once per shape (resident vertex/index runs), with one compact
  stream op per packet instead of material display lists, vertex-format lists, CP array writes, indexed
  matrix loads and draw lists. Everything else (particles, the sea, shadows, EFB copies, goop, 2D) stays
  on the GX emulation path, in the same ordered stream.
- **Where the time goes today.** The GX thread on core 1 is busy 39 ms a frame, and roughly half of that
  (~20 ms, estimate) is J3D: display-list calls and indexed matrix loads. On the game thread, the GX list
  and GX post phases take 20.7 ms, but J3D draw buffers are only ~5.6 ms of that and ~60 KB of the 236 KB
  stream; most of those phases' time goes to direct-GX objects and draw-cue logic, which this does not
  touch.
- **Expected savings (estimates).** GX thread −17 to −21 ms (to ~20 ms); game thread −6 to −9 ms (J3D
  draw work, the 3.6 ms frame-end wait for the GX thread, later the per-frame material list rebuilds);
  render thread −2 to −4 ms. **GPU: nothing.** It stays at ~40 ms (vertex/triangle bound), so native J3D
  alone gives ~25 fps, not 30: it takes the GX thread out of the race and leaves the GPU and the game
  thread as the limits.
- **First steps.** Stage 0 measures each draw buffer's GX-thread cost (no behaviour change). Stage 1 draws
  `DrawBuf MapOpa`'s locked materials natively: small in savings, but it builds and proves every piece
  (shape compile, material compile, the new record, state hand-over, parity check) on static geometry.

## 1. The path today

```
game thread (core 0)              GX thread (core 1)                  render thread (core 2)       GPU
J3DDrawBuffer::draw               parses the stream                   applies state deltas (rs)
 J3DMatPacket::draw                BP/CP/XF -> register file g        readTev, TEV cache key/lookup
  material DL call                 DL cache: validate, replay the     translateTev on a miss
 J3DShapePacket::draw               state-only lists, draw resident   drawGpu: uniforms from rs.xfMem
  vcd/vat DL, CP array bases,       runs (arena)                      pica_draw_gx (citro3d)
  matrix-index loads, draw DLs     indexed XF loads read the matrices
GX API: write filter, stream       batches + state deltas -> ring
```

Measured, profile window:

| Where | ms/frame | Notes |
|---|---|---|
| Frame | 51–55 | 41–51 in the same run's Plaza frames outside profile runs |
| Game thread phases | movement 15.9, calc-anim 6.2, last-tick 4.0, **GX list 7.6**, **GX post 13.1**, end rendering 3.7, hit check 1.9 | end rendering includes 3.6 ms waiting at the display copy for the GX thread |
| Game thread stream writes | GX list 100 KB, GX post 136 KB | the profiler prices the writes themselves at 1.9 + 2.7 ms |
| GX thread | busy 39.2 | 31–43 in normal Plaza frames |
| GX thread by command (sampled 1 in 16; sums to 48.5, so read as shares) | list calls 893/16.4, list copies 42/1.7, indexed XF 1739/6.0, BP 9065/11.1, CP 4251/4.2, XF 1906/3.6, primitives 262/2.5, other 3.0 | |
| GX thread, display lists | validate 3.4 + replay 6.1 + uncached 1.6; direct commands 18.2 | 936 calls/frame, 904 cached |
| Render thread | busy 28–33 (normal Plaza frames) | tev 6.1 (read 1.0, key + lookup 3.8, translate 0.3), submit 9.0, xform 2.6, tex 1.5 |
| GPU | 40.5 | vertex 13.0, pixels 11.7, setup + near clip 5.5, past the sides 3.4, rest 6.9 |

Who writes the GX list and GX post phases (top rows of the per-object list, game-thread exclusive ms,
stream KB):

| Object | Phase | ms | KB | Verts | GPU ms (left out) |
|---|---|---|---|---|---|
| TLightWithDBSetManager (lit J3D buffers) | post / list | 2.86 / 0.85 | 32.9 / 9.4 | 11850 / 6949 | 3.59 / 1.52 |
| TMapObjWave (the sea) | post | 2.04 | 42.7 | 1352 | 2.88 |
| TMarioParticleManager (JPA) | post | 1.89 | 18.7 | 634 | 1.60 |
| TLightCommon, TLightMario (light loads) | list + post | 2.26 | 19.6 | 0 | – |
| TMBindShadowManager (GD shadow lists) | list | 1.31 | 26.2 | 18704 | 2.09 |
| TMirrorActor (mirror entry) | list | 0.99 | 0.1 | 0 | – |
| DrawBuf StaticMapObj SunOpa (J3D, SDL) | list | 0.74 | 9.9 | 5095 | 0.45 |
| TGCConsole2 (HUD, J2D) | post | 0.56 | 4.5 | 99 | – |
| DrawBuf MapOpa (J3D) | list | 0.49 | 5.2 | 64875 | 9.94 |
| TReInitGX | post | 0.42 | 3.6 | 0 | – |
| TModelWaterManager (Shine shadow shells) | list | 0.33 | 4.9 | 1916 | 4.76 |
| Other J3D draw buffers (Mirror, MapXlu, ChrOpa, Sky, Graffito, AfterIndirect) | both | ~0.7 | small | ~10k | ~8 |

So of the 20.7 ms: J3D draw buffers ~5.6 ms and ~60 KB; direct-GX objects ~9 ms and ~120 KB; the rest
is draw-cue logic and objects below the list's cut. The map is cheap on the game thread already (resident
runs, locked materials); its cost is on the GPU.

## 2. What J3D sends, packet by packet

References are to `build/decomp-patched/libs/JSystem/src/J3D/` unless a path is given.

### 2.1 Entry (last-tick list, phase 8)

`TMarDirector::preEntry` (`src/System/MarDirectorPreEntry.cpp`) lists calc-view and entry for every group
after setting each `DrawBuf` as j3dSys's draw buffer. Per model, `J3DModel::viewCalc`
(`J3DGraphAnimator/J3DModel.cpp:1068`) swaps the double-buffered draw and normal matrix arrays and fills
the new ones (view × joint, envelopes, inverse transposes, billboards), then `prepareShapePackets` points
each shape packet at them and at the vertex buffer's current arrays. `J3DJoint::entryIn`
(`J3DGraphAnimator/J3DJoint.cpp:351`), for each visible material on the joint:

1. unless the packet is locked: material animation (`J3DMaterialAnm::calc`: material colour, texture
   number, TEV colour and konst colour, texture SRT), texture-matrix calc (`J3DTexGenBlockBasic::calc`,
   `J3DGraphBase/J3DMaterial.cpp:845`: env, proj and view-proj maps use the view and joint matrices), and
   `setCurrentMtx` (the shape's texture-matrix slots);
2. `J3DMatPacket::entry` into the opaque or translucent buffer by its sort mode (material, material anim,
   Z, model, none: `J3DGraphBase/J3DDrawBuffer.cpp:55`). Material sort merges packets with the same
   material ID into one material packet with several shape packets;
3. if the entry made a new packet, `J3DMaterial::makeDisplayList` (`J3DMaterial.cpp:808`). Unless the
   packet is locked, this swaps the packet's two DL buffers and writes, with GD: TEV block (texture image
   registers via `loadTexNo`, TEV orders, texture-coordinate scale, TEV colour and konst registers, stages,
   konst selection and swap tables), indirect block, PE block (alpha compare, blend, Z mode, Z comp loc,
   fog, dither), gen mode, colour block (material and ambient colours, channel controls, light objects for
   lit materials), texgen block (texgens and texture matrices).

Locking: `MActor::initDL` (`src/M3DUtil/MActor.cpp:250`) builds and locks the lists of materials without
material animations; `TJointModel::initActor` (`src/Map/JointModel.cpp`) and `SMS_MakeDLAndLock`
(`src/MarioUtil/DrawUtil.cpp`) lock whole models. A locked packet's list is built once. With model flag
`0x20000` instances share one list per material (`newSharedDisplayList`), rewritten by each instance.

### 2.2 Draw (GX list phase 10, GX post phase 12)

`TDrawBufObj::perform(CUE_DRAW)` calls `J3DDrawBuffer::draw` (head or tail order), which calls each
packet's `draw`:

- `J3DMatPacket::draw` (`J3DGraphBase/J3DPacket.cpp:140`): returns if every shape packet is hidden;
  sets j3dSys's texture table and packet; `J3DMaterial::load` calls the packet's material list
  (`GXCallDisplayList`), unless j3dSys flag 2 is set (shape-only drawing with state the caller set:
  `TJ3DSysFlag`, Mario's silhouette and logic draws); then each shape packet's `draw`.
- `J3DShapePacket::draw` (`J3DPacket.cpp:171`): if visible, callback phase 0 (`src/MarioUtil/PacketUtil.cpp:234`:
  TEV colour, konst colour or material colour overrides, fog off, an extra list), sets vertex arrays and
  matrix arrays for the view number, `J3DShape::draw`, callback phase 1.
- `J3DShape::draw` (`J3DGraphBase/J3DShape.cpp:227`):
  - the shape's vertex-format list (`mVcdVatCmd`, 0xC0 bytes: VCD, array bases and strides, VAT 0),
    built once at load (`makeVcdVatCmd`);
  - `J3DLoadArrayBasePtr` for position, normal (unless NBT) and colour 0: CP writes, the current arrays
    (a deformed model's change every frame);
  - `GXSetArray` for the position and normal matrix arrays: CP writes, the model's current draw and
    normal matrix arrays for this view;
  - `JRNLoadCurrentMtx`: CP and XF matrix-index registers (texture matrix slots from the material);
  - per matrix group: `J3DShapeMtx::load` (one matrix: `GXLoadPosMtxIndx` and `GXLoadNrmMtxIndx3x3`;
    `J3DShapeMtxMulti`: up to 10, slot by slot; CPU-skinned pipelines load the view matrix instead;
    `J3DShapeMtxDL`: a 32-byte list), then `J3DShapeDraw::draw`, the group's draw list (indexed
    attributes, per-vertex matrix indices when the group has several matrices).

On the GX thread each list call goes through the display-list cache (`gx_fifo.cpp:1527`): state-only
lists (materials, vertex formats) are decoded once into register ops and replayed; draw lists become
resident runs (`gx3ds_backend.cpp:3956`), validated on each call by write stamps and hashes. Each indexed
load reads 12 or 9 words from the game's matrix array (`loadIndexedXF`, `gx_fifo.cpp:1032`).

### 2.3 Fixed from load vs per frame

| Data | Lives in | Changes |
|---|---|---|
| Vertex attribute arrays | `J3DVertexData` | fixed, except deformed models (`J3DVertexBuffer` current arrays: CPU skinning on some bosses, cluster deform, vertex-colour calc) |
| Draw lists, vertex-format list, matrix-index tables | `J3DShape`, `J3DShapeDraw`, `J3DShapeMtx*` | fixed |
| Draw and normal matrices | `J3DModel` `mDrawMtxBuf`/`mNrmMtxBuf[viewNo]`, double-buffered | every frame |
| TEV, PE, gen mode, channel-control and texgen structure | material blocks | fixed in practice; game code changes some at init (`TMapModel::initUnderpass`) or through `change`/`copy` |
| Material, ambient, TEV and konst colours | material blocks, `J3DMaterialAnm` | animated |
| Texture number (image address, size, format) | TEV block `mTexNo`, `J3DTexNoAnm` | animated |
| Texture matrices | `J3DTexMtx::mTotalMtx` | SRT animation; env/proj maps every frame (view and joint) |
| Visibility | shape flag 1, `J3DShapePacket::hide`/`show` | per frame |
| Draw order | `J3DDrawBuffer` sort | per frame |
| Lights, ambient for light-off materials, projection, viewport, scissor, PE masks, fog | outside J3D (light manager, camera, `drawInit`, `TReInitGX`) | per buffer, per frame |

### 2.4 Two facts the design has to respect

- **A packet's material state exists only in its list.** Instances of a model share `J3DMaterial`
  objects; `entryIn` runs each instance's animation on the shared material and then writes that
  instance's list. By draw time the material object holds the last instance's values. A native path must
  take its snapshot where `makeDisplayList` runs, per list buffer, not read the material at draw time.
  Keying snapshots by list buffer also reproduces shared lists (`0x20000`) exactly: whatever buffer
  `callDL` would have called, the snapshot of that buffer is used.
- **Order is decided in game code.** Hooking below `J3DDrawBuffer::draw` (at the packet) keeps the sort
  modes, Z sort, SDL instancing (`src/M3DUtil/SDLModel.cpp`: one material packet, every instance's shape
  packet) and mirror views exactly as they are.

## 3. The native path

### 3.1 Overview

At the packet, the game thread sends a stream op instead of GX calls. The GX thread, which already owns
the decoder, the arena and the register file, turns it into a render record without parsing, compiling
shapes and materials the first time it sees them. The render thread draws the record with
`pica_draw_gx` (`platform/3ds/pica/pica_draw.c:1285`), unchanged. A packet the path cannot express falls
back to today's code, in place, in order.

### 3.2 Shapes: compiled once into resident runs

Key: the `J3DShape*` plus the vertex array pointers it was compiled against (the vertex-format list and
draw lists are immutable after load).

Compile, on the GX thread at first use: parse `mVcdVatCmd` into a scratch register file, then decode each
`J3DShapeDraw` list with the same code the display-list cache uses (`dlRecordPrimitive` and the
attribute readers in `gx_fifo.cpp`, byte order included) into packed runs, and store them with
`arenaUpload` (`gx3ds_backend.cpp:4211`): duplicates merged, triangles reordered when the shape's
material allows it (`orderWhy`; a J3D shape has one material, but game code can switch it to blending at
run time, so the draw-time check the arena keeps today stays), bounds kept for culling. Per group: its matrix slots (`J3DShapeMtx` index or the `J3DShapeMtxMulti` table) and whether
vertices carry matrix indices. The compiled shape owns these runs; the display-list cache never sees the
lists, so nothing is stored twice.

Not compiled (fall back): shapes whose current vertex arrays are not the model data's (deformed models),
`J3DShapeMtxDL` groups until decoded, NBT shapes until checked.

### 3.3 Materials: compiled into a PICA state block

Input: the material's register image, which is exactly what its list writes. Two ways to get it:

- **Locked packets (stage 1).** Parse the packet's list once into a scratch register file (the
  state-only list decoder, `dlCompileState`). The list cannot change while locked; `__wrap_GDInitGDLObj`
  (`gx3ds_backend.cpp:4962`) already stamps every rewrite, so validation is one stamp compare, not a hash.
- **Rebuilt packets (stage 3).** Patch `J3DMaterial::makeDisplayList` to also write a compact snapshot
  next to the list buffer: a structure hash over the fields that build the list, plus the values that
  animate (colours, texture images, texture matrices). On a structure the GX thread has not seen, the
  game thread sends the list bytes once (the existing list-copy op); otherwise only the hash and values.

Output: the backend's existing translation, run on that image instead of on `rs`: `readTev`
(`gx3ds_backend.cpp:1581`), `translateTev`/`translateTevFallback` (1666, 2207), `currentPicaState` (651)
and `alphaTest`. These read the global render-side registers today; they need a register-file argument,
a mechanical change. The result is a `PicaState` template (combiner stages, blend, depth, alpha test,
cull, texture-unit wiring and wrap/filter) plus how to fill the per-draw parts.

Two subtleties carried over from today's TEV cache (`tevCacheKey`, 2698):

- `translateTev` folds TEV registers known at translation time into constants. A material whose TEV
  colours animate changes its translation with the values, so the compiled-state cache is keyed by
  (material structure, the folded values), as today's key includes them. Static materials hit one entry.
- Part of the state is global: scissor, colour and alpha update, destination alpha, fog, lights and
  ambient for light-off materials, and the masked BP writes the list merges into. The GX thread keeps a
  *global epoch*, bumped by any write to that register set; the compiled state is keyed by (material,
  epoch). Globals change a few times a frame (`drawInit`, silhouettes), so hits stay near 100 %.

Today the render thread builds a 70-register key and hashes the `PicaState` for every draw (3.8 ms a
frame); a compiled material replaces that with a pointer and an epoch compare.

### 3.4 Per-frame updates, without GX commands

| What | Comes from | Goes to |
|---|---|---|
| Position/normal matrices | the model's current arrays and the group's slots (CPU-skinned pipelines: the view matrix) | uniform rows; copied by the GX thread |
| Material and ambient colour | the packet's snapshot | `CS`/`CM` uniforms |
| TEV and konst colours | snapshot | combiner constants; translation key when folded |
| Texture images | snapshot (`ResTIMG` address, size, format, TLUT) | the render thread's texture cache, keyed as today (EFB-copy textures included) |
| Texture matrices | snapshot (`mTotalMtx`), shape's matrix slots | `T` rows, composed as `drawGpu` does (post matrix, EFB-copy rectangle) |
| Visibility | `isVisible`, shape flag 1, hidden material packets | nothing sent |
| View number (mirror) | `mpCurrentViewNo` | which matrix array |
| Culling | compiled bounds × matrix | skip, as `dlRunOffscreen` does today |
| Profiler | current object, cut flag | the record (per-object vertices, GPU-by-object cuts) |

No game-side maths changes: J3D's matrix code (exact Gekko arithmetic) runs as now; the native path copies
the floats the indexed loads read today. Uniform composition is the same float code `drawGpu` uses.

### 3.5 Where the work runs, and the op

The game thread is the critical path, so it does the least: one op per material packet (snapshot
reference) and one per visible shape packet, about 10 words each (shape, view's matrix arrays, vertex
arrays, pipeline, texture-matrix slots, flags). New stream ops next to `kStreamOpCallDL`
(`gx_internal.h:328`), for example `kStreamOpJ3DMat = 0xCC` and `kStreamOpJ3DShape = 0xCD`. The game thread
calls `apiForget(7)` at a material op, as `GXCallDisplayList` does (`gx_api.cpp:610`), so the write
filter's knowledge stays what it is today.

The GX thread resolves matrix pointers (allowed for the same reason indexed loads are: the game waits at
each display copy until the GX thread has finished that frame, 1c38c10), culls, and writes a new render
record `RC_J3D` (next to `RC_STATE`/`RC_DRAW`, `gx3ds_backend.cpp:3357`) with the compiled shape and
material, dynamic values and matrix rows. It does not touch `g` (see 4.3).

The render thread draws `RC_J3D` with a small `drawJ3D`: textures from the cache, uniforms from the
record, `pica_draw_gx` with the compiled `PicaState`, uploading only the matrix rows the group uses (today:
2599 rows a frame).

## 4. What stays on the GX emulation path, and sharing state

### 4.1 Direct GX users (unchanged)

| User | Why it stays | Plaza cost |
|---|---|---|
| JParticle emitters (`JPADrawVisitor.cpp`, TMarioParticleManager) | immediate-mode quads per particle | 1.89 ms game (+1.70 calc), 18.7 KB, GPU 1.6 |
| TMapObjWave (`src/MoveBG/MapObjWave.cpp`) | immediate-mode grid rebuilt every frame | 2.04 ms, 42.7 KB, GPU 2.9 |
| TMBindShadowManager (`src/MarioUtil/ShadowUtil.cpp`) | GD-built shadow volumes, destination-alpha tricks | 1.31 ms, 26.2 KB, GPU 2.1 |
| TModelWaterManager (`src/Player/ModelWaterManager.cpp`) | spray and the Shine shadow shells, GD lists | 0.33 ms, GPU 4.8 |
| EFB copies (`JDREfbCtrl.cpp`, BathWaterManager, hx_wiper, bossManta) and the display copy | copies and their readbacks | – |
| Goop (`src/Map/Pollution*.cpp`) | stamps, copies, readbacks for counting | 0.73 ms |
| 2D: J2D, JUTFont, GC2D (console, fader, talk, sunglasses, wiper) | immediate mode | 0.56 ms (console) |
| Lights and global state: TLightCommon/TLightMario, `J3DSys::drawInit`, `TReInitGX`, `TJ3DSysFlag` | global registers the native path reads | 2.7 ms |
| Others with GXBegin or GD: MapWire, grass, flags, poles, pool, MapDraw walls, beam, BathWater | immediate mode | TMapObjFlagManager 0.20 ms |

### 4.2 J3D packets that fall back

A packet is drawn the old way, in place, when any of these holds (each can be removed in a later stage):
j3dSys flag 2 (shape-only drawing with the caller's state); a shape-packet callback; a deformed vertex
buffer; a material the compile has not been taught (indirect stages, NBT scale); an unlocked material in
stages 1–2; a draw buffer not on the enabled list.

### 4.3 Sharing state safely

- **One ordered channel.** Native ops are bytes in the same stream, so the GX thread sees them in program
  order between emulated commands, and the render ring keeps that order. No second producer.
- **Three register copies.** The API's copy on the game thread (`GA`, the write filter's), the GX thread's
  `g`, and the render thread's `rs`. A native material op forgets the filter's knowledge as a list call
  does. `rs` keeps receiving state deltas for emulated draws; native records carry their own state and
  leave `rs` alone.
- **Leftover state, materialised lazily.** On the console, a draw after J3D packets inherits the last
  material's and shape's registers. The GX thread remembers the last native material image and shape
  effects (VCD/VAT, array bases, matrix index registers, the loaded matrices) and writes them into `g`,
  with `bpTouched`/`xfTouched`, just before the next emulated command that can read state (register
  write, list call, primitive, copy). Exact, and paid once per native-to-emulated transition, not per
  packet. Stage 0 can count the transitions.
- **Global state into native draws.** Global registers are written by emulated commands into `g`; the
  epoch (3.3) makes compiled states follow them, and their values reach the render thread through the
  usual state deltas, in order.
- **Lifetimes.** Matrices and vertex arrays are read by the GX thread during the frame (the existing
  frame-end wait), and copied into records for the render thread. Snapshots live with their list buffers,
  which J3D double-buffers. Compiled shapes and materials are keyed by pointers into stage heaps, so the
  caches are cleared at scene change (where the arena and display-list cache already reset), and keyed
  with a generation so a freed-and-reused address cannot match.
- **Textures.** The render thread's texture cache stays the only owner; native records carry the same keys
  emulated draws do (`kStreamOpTexImage`'s host pointers), so EFB-copy textures and write-stamped
  invalidation work unchanged.

## 5. Expected savings

Estimates, from the measurements in section 1:

| | Now | J3D's part | After all J3D is native |
|---|---|---|---|
| GX thread (core 1) | 39.2 ms | list calls and copies (~85–95 % J3D), all 1739 indexed loads, most of the CP writes (the redundant direct CP writes are led by J3D's: position and colour array bases, matrix-array strides, matrix-index registers), some XF: ~52–58 % of the sampled total, **~20–23 ms** | ~18–22 ms (native ops ~1–2 ms at an estimated 2–4 µs each for ~400 shape packets) |
| Game thread (core 0) | ~47–55 ms profiled; GX list + post 20.7 | J3D draw buffers ~5.6 ms (GX API calls, filter, list-call bookkeeping per 32-byte block, plus packet traversal that stays); frame-end wait 3.6 ms; list rebuilds in entry (part of last-tick 4.0, 24.7 KB of list copies a frame) | −3 to −4 (draws) −3.6 (the wait, once the GX thread finishes first) −0.5 to −1.5 (rebuilds, stage 4) = **−6 to −9 ms** |
| Render thread (core 2) | 28–33 ms | TEV read, key and lookup 4.8 ms, state-delta decode | −2 to −4 ms (not on the critical path) |
| GPU | 40.5 ms | – | **unchanged** |

The ~20 ms and ~240 KB of the GX phases do not translate into J3D savings: J3D draw buffers are ~5.6 ms
and ~60 KB of them, and the stream writes themselves are priced at 4.6 ms.

Frame: today about max(game ~43–47, GX ~39 plus the wait, GPU ~40) ≈ 45 ms. Afterwards about
max(game ~37–40, GX ~20, GPU ~40) ≈ 40 ms, about 25 fps. 30 fps then needs the GPU below 33 ms (fewer
vertices or cheaper ones: Plaza is vertex/triangle bound, and native 400×240, chunk culling, guard band
and strips have been measured as dead ends) and the game thread below 33 ms (movement list 15.9 ms, the
direct-GX objects of 4.1).

What the compile makes possible on the GPU (stage 5, unmeasured): per-material vertex shader variants
(today one shader branches on bool uniforms: 203 a frame), dropping attributes a material never reads,
larger palettes so multi-group shapes draw once, culling skinned shapes by their bounds.

## 6. Staged plan

Each stage is behind a setting and leaves every other path as it is.

**Stage 0: attribution (no behaviour change).** Credit GX-thread time, not only vertices, to the object
the profiler marks (`kStreamOpProfObj`, patch 0028), and split it by origin: J3D material list, J3D
vertex-format list, J3D draw list, J3D indexed load and CP, other. Count native-to-emulated transitions
(J3D packet followed by a direct command). One hardware run gives each draw buffer's ceiling and decides
whether stage 3 should come before stage 2.

**Stage 1: `DrawBuf MapOpa`, locked materials.** Single- and multi-matrix static shapes, locked lists, no
callbacks, flag 2 off. Pieces: the decomp patch at `J3DMatPacket::draw`/`J3DShapePacket::draw` (and a
per-buffer switch at `J3DDrawBuffer::draw`, resolved by name at scene setup as
`MarDirectorSetupObjects.cpp` already does for the sky buffers); the two stream ops; shape and material
compile on the GX thread; `RC_J3D` and `drawJ3D`; lazy materialisation; the check mode (section 8).
Expected: small (1–4 ms GX thread, estimate; the map's game-thread cost is 0.49 ms). Exit: check mode
reports no mismatch, GPU time unchanged, GX thread and indexed-load counts down by MapOpa's share.

**Stage 2: the other static buffers.** MapXlu (blended: reordering stays off by the same rule), Sky
Opa/Xlu, StaticMapObj Sun/Shadow Opa/Xlu (SDL instancing: one material op, many shape ops), Mirror
Opa/Xlu (view number), AfterIndirect (EFB-copy textures).

**Stage 3: animated models.** The light manager's buffers and ChrOpa/ChrXlu: unlocked materials through
the `makeDisplayList` snapshot, envelope and multi-matrix shapes, texture SRT and env/proj maps, texture
pattern and TEV colour animation (translation keyed by folded values). The largest share of both the GX
and game-thread savings is expected here (TLightWithDBSetManager alone is 3.7 ms of game thread).

**Stage 4: the remainder.** Shape-packet callbacks as per-draw overrides (TEV, konst and material colour;
the fog-off and list-call cases keep falling back), flag-2 shape-only draws (native geometry with the
material state from `g`), deformed vertex buffers (per-frame upload into the ring), and skipping
`makeDisplayList`'s GD writes for packets only ever drawn natively.

**Stage 5 (optional): GPU.** The items at the end of section 5, each with its own A/B.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Visual parity. The reference is the current 3DS output (its one fixed light, no fog, no indirect), not the GameCube. | The same translation code runs on the same register image; the check mode compares every draw; fall back per packet on anything not understood. |
| Per-instance material state (2.4) | Snapshot per list buffer at `makeDisplayList`; stage 1 takes locked lists only. |
| Material animation changes the translation (folded constants) | Key compiled state by the folded values, as the TEV cache does; misses cost what they cost today. |
| Texture matrices depend on view and joint every frame | They come from the snapshot (`mTotalMtx` after `calc`), never cached as static. |
| State leaking between the two paths | Lazy materialisation into `g`; filter forget at material ops; global epoch. |
| Draw order | Hook at the packet, below the draw buffer's sort; nothing reorders packets. |
| Linear memory (5.3–6 MB free in Plaza; arena capped at 8 MB) | Compiled shapes own their runs instead of the display-list cache; nothing is stored twice. |
| First-use hitches | Compile on first draw, as the display-list cache records today; precompile at model load later if it shows. |
| Stale compiled data after a scene change | Clear at scene change with the arena; generation-tagged keys. |
| Tooling regressions (trace, debug colours, profile cuts, culling counters) | `RC_J3D` carries the profile object and cut flag; trace and debug colours get a `drawJ3D` branch in stage 1. |
| The GX thread is on the slow core | The point of the change is to take work off it; nothing new is moved there except op forwarding. |

## 8. A/B on hardware

The user runs every build on a New 3DS; builds are staged in the Azahar sdmc folder only.

- **Builds.** One app per case, each in its own `/3ds/<name>/<name>.3dsx` folder with its own title:
  A = native off, B = native on for the stage's buffers, C = check mode. Delete `build3ds/main.o`,
  `sms3ds.smdh` and `sms3ds.3dsx` between builds and confirm the build stamps differ. The mode is a
  `settings.txt` key (for example `native_j3d = off | on | check` and `native_j3d_buffers = MapOpa`) whose
  default differs per build, so a build can also be flipped on the card.
- **Runs.** The same spot in Delfino Plaza, standing still, a profile run (bottom-right touch) in each.
- **Compare** (grep the logs; never read them whole): GX thread busy and the by-command line (list calls
  and indexed XF should fall by the buffers' share), display lists per frame, game-thread GX list/post
  phases and the frame-end wait, render-thread tev, GPU ms (should match within ~1 ms) and frame ms.
- **Check mode.** The game thread sends both the native ops and the old GX calls. The emulated draws are
  what is drawn; the render thread compares, per packet, the native record's `PicaState` (bytewise, as the
  PICA side compares states), uniforms, textures and index ranges with the emulated batches', and logs
  each mismatch once with the model and material name plus a count in `[perf]`. Exit criterion for every
  stage: zero mismatches over a walk through the stage's areas.
- **Second check.** One-frame traces (L) from builds A and B at the same spot, diffed by script: same
  draws, same states. Useful for static buffers (stage 1–2); moving models make frames differ.

## Appendix A: how the numbers were read

- Log: `sdmc/3ds/log-notv.txt` in the Azahar staging folder, build Oct 19 2026 15:17:47, one profile run
  in area 1-0 (Delfino Plaza). The sibling `log-v.txt` (15:17:45) gives the same picture (GX list 7.4,
  GX post 14.3, GPU 36.9 ms).
- Object names: the log's vtable addresses sit 0x13C0 below the same vtables in the current
  `sms3ds.elf` (checked on TDrawBufObj, TLightWithDBSetManager, TMapObjWave and TMarioParticleManager), so
  `scripts/prof_names.py` with that offset added named the rows.
- The "by command" line times 1 in 16 commands and adds per-command overhead: its total (48.5 ms) exceeds
  the thread's busy time (39.2 ms), so it is used for shares, scaled to busy time.
- J3D shares of GX-thread time are estimates: list calls are taken as 85–95 % J3D (the GD lists of
  shadows, water and callbacks are the rest), indexed loads as 100 % (only `J3DSys::loadPosMtxIndx` and
  `loadNrmMtxIndx` issue them), CP writes as mostly J3D from the per-register counts. Stage 0 replaces
  these with measurements.

## Appendix B: code map

| Piece | Where |
|---|---|
| Packet draws | `J3DGraphBase/J3DPacket.cpp:140,171`; shape draw `J3DShape.cpp:227`; material list `J3DMaterial.cpp:808,829` |
| Entry and list rebuilds | `J3DGraphAnimator/J3DJoint.cpp:351`; `J3DModel.cpp:705,1068`; `src/M3DUtil/MActor.cpp:250` |
| Draw buffers | `J3DGraphBase/J3DDrawBuffer.cpp`; `JDrama/JDRDrawBufObj.cpp`; cue lists `src/System/MarDirectorPreEntry.cpp` |
| GX API, write filter, list calls | `build/pcport-3ds/platform/gx/src/gx_api.cpp:106,610,878` |
| Parser, display-list cache, indexed loads, stream ops | `gx_fifo.cpp:1032,1237,1370,1527,1817`; `gx_internal.h:328` |
| TEV translation and cache | `platform/3ds/gx/gx3ds_backend.cpp:651,1581,1666,2698` |
| GPU vertex path | `gx3ds_backend.cpp:2741` (`drawGpu`); `platform/3ds/pica/pica_draw.c:1285` |
| Render ring, records, state deltas | `gx3ds_backend.cpp:3357-3830` |
| Resident runs | `gx3ds_backend.cpp:3956,4211` |
| GX thread, list rewrites | `gx3ds_backend.cpp:4392,4481,4547,4962` |
