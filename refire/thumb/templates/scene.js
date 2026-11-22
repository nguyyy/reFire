// The 3D world behind zy0x and the hook: a background plate, floating cards in perspective
// with soft shadows and rim glow, light rays, sparkles, bloom. Built only from the
// <script id="scene-spec"> json (python decides every value; refire.thumb.render serves this
// page from http://refire.local so module imports resolve to the vendored three.js).
// Coordinates are css pixels: at z = 0 one unit is one pixel, origin top-left, y down.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const S = JSON.parse(document.getElementById('scene-spec').textContent);
const W = S.w, H = S.h, D2R = Math.PI / 180;

const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('scene'), antialias: true,
                                           preserveDrawingBuffer: true, alpha: false });
renderer.setPixelRatio(S.dpr || 1);
renderer.setSize(W, H, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x000000, 1);

const scene = new THREE.Scene();
const fov = S.fov || 30;
const dist = (H / 2) / Math.tan(fov / 2 * D2R);
const camera = new THREE.PerspectiveCamera(fov, W / H, 1, dist * 6);
camera.position.set(W / 2, -H / 2, dist);
camera.lookAt(W / 2, -H / 2, 0);

// seeded rng: same spec -> same sparkles -> same bytes
let seed = (S.seed || 1) >>> 0;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

const loader = new THREE.TextureLoader();
const tex = (src) => new Promise((ok, bad) => loader.load(src, (t) => {
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; ok(t);
}, undefined, bad));

// a plane whose top-left sits at css (x, y) at depth z (z < 0 = further away, drawn scaled up
// so `w` still means pixels on screen at z = 0 when `screen` is set)
function place(mesh, { x, y, w, h, z = 0, rot = [0, 0, 0] }) {
  mesh.position.set(x + w / 2, -(y + h / 2), z);
  mesh.rotation.set(rot[0] * D2R, rot[1] * D2R, rot[2] * D2R, 'YXZ');
  return mesh;
}

function softRect(w, h, blur, color = '#000') {         // blurred rounded rect, for shadows + glows
  const pad = blur * 3, c = document.createElement('canvas');
  c.width = Math.ceil(w + pad * 2); c.height = Math.ceil(h + pad * 2);
  const g = c.getContext('2d');
  g.filter = `blur(${blur}px)`; g.fillStyle = color; g.fillRect(pad, pad, w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return { t, pad };
}

async function background(bg) {
  const t = await tex(bg.src);
  const z = bg.z ?? -400;
  const k = (dist - z) / dist;                           // grow so it still covers the frame
  const m = new THREE.MeshBasicMaterial({ map: t, color: new THREE.Color(bg.tint ?? 1, bg.tint ?? 1, bg.tint ?? 1) });
  const p = new THREE.Mesh(new THREE.PlaneGeometry(W * k * (bg.scale ?? 1.08), H * k * (bg.scale ?? 1.08)), m);
  p.position.set(W / 2, -H / 2, z);
  scene.add(p);
}

async function card(c) {
  const t = await tex(c.src);
  const h = c.h ?? c.w * t.image.height / t.image.width;
  const g = new THREE.Group();
  if (c.shadow) {                                        // soft shadow, offset down-right behind the card
    const { t: st, pad } = softRect(c.w, h, c.shadow_blur ?? 24);
    const sm = new THREE.Mesh(new THREE.PlaneGeometry(c.w + pad * 2, h + pad * 2),
      new THREE.MeshBasicMaterial({ map: st, transparent: true, opacity: c.shadow, depthWrite: false }));
    sm.position.set(c.shadow_dx ?? 14, -(c.shadow_dy ?? 18), -6);
    g.add(sm);
  }
  if (c.glow) {                                          // rim glow just outside the edges
    const { t: gt, pad } = softRect(c.w, h, c.glow_blur ?? 10, c.glow);
    const gm = new THREE.Mesh(new THREE.PlaneGeometry(c.w + pad * 2, h + pad * 2),
      new THREE.MeshBasicMaterial({ map: gt, transparent: true, opacity: c.glow_alpha ?? 0.8,
                                    blending: THREE.AdditiveBlending, depthWrite: false }));
    gm.position.z = -3;
    g.add(gm);
  }
  const face = new THREE.Mesh(new THREE.PlaneGeometry(c.w, h),
    new THREE.MeshBasicMaterial({ map: t, transparent: true, color: new THREE.Color(c.tint ?? 1, c.tint ?? 1, c.tint ?? 1) }));
  g.add(face);
  scene.add(place(g, { ...c, h }));
}

function rays(r) {                                       // light rays fanning out of a point behind zy0x
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uO: { value: new THREE.Vector2(r.x, r.y) }, uC: { value: new THREE.Color(r.color ?? '#ffffff') },
                uS: { value: r.strength ?? 0.35 }, uN: { value: r.count ?? 18 }, uR: { value: Math.max(W, H) * (r.reach ?? 0.9) },
                uSeed: { value: S.seed || 1 } },
    vertexShader: 'varying vec2 vP; void main(){ vP = vec2(position.x, -position.y); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `uniform vec2 uO; uniform vec3 uC; uniform float uS, uN, uR, uSeed; varying vec2 vP;
      float h(float n){ return fract(sin(n * 12.9898 + uSeed) * 43758.5453); }
      void main(){ vec2 d = vP - uO; float a = atan(d.y, d.x) / 6.2831853 + 0.5; float i = floor(a * uN);
        float f = fract(a * uN); float w = 0.25 + 0.5 * h(i); float beam = smoothstep(w, w - 0.18, abs(f - 0.5) * 2.0);
        float fall = 1.0 - smoothstep(0.0, uR, length(d)); gl_FragColor = vec4(uC * beam * fall * uS * (0.5 + h(i + 7.0)), 1.0); }`,
  });
  const p = new THREE.Mesh(new THREE.PlaneGeometry(W * 1.6, H * 1.6), mat);
  p.position.set(W / 2, -H / 2, r.z ?? -60);
  // the shader works in css px of the z = 0 plane: undo the plane's own depth scaling
  const k = (dist - (r.z ?? -60)) / dist;
  p.scale.set(k, k, 1);
  mat.uniforms.uO.value.set((r.x - W / 2) / k + W / 2, (r.y - H / 2) / k + H / 2);
  scene.add(p);
}

async function overlay(o) {                            // e.g. speed lines: centered on a point, behind the cards
  const t = await tex(o.src);
  const z = o.z ?? -150, k = (dist - z) / dist;          // depth-scaled so `w` is screen pixels
  const w = o.w * k, h = w * t.image.height / t.image.width;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: t, transparent: true,
    opacity: o.opacity ?? 0.5, depthWrite: false }));
  m.position.set((o.cx - W / 2) * k + W / 2, -((o.cy - H / 2) * k + H / 2), z);
  scene.add(m);
}

function sparkles(s) {
  const n = s.count ?? 40, pos = new Float32Array(n * 3), c = document.createElement('canvas');
  for (let i = 0; i < n; i++) {
    pos[i * 3] = (s.x0 ?? 0) + rnd() * ((s.x1 ?? W) - (s.x0 ?? 0));
    pos[i * 3 + 1] = -((s.y0 ?? 0) + rnd() * ((s.y1 ?? H) - (s.y0 ?? 0)));
    pos[i * 3 + 2] = -rnd() * 80;
  }
  c.width = c.height = 64;
  const g = c.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.25, 'rgba(255,255,255,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: s.size ?? 10, map: new THREE.CanvasTexture(c),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, color: new THREE.Color(s.color ?? '#fff6d8') })));
}

async function build() {
  if (S.bg) await background(S.bg);
  if (S.rays) rays(S.rays);
  for (const o of S.overlays || []) await overlay(o);
  for (const c of S.cards || []) await card(c);         // in order: later cards sit on top
  if (S.sparkles) sparkles(S.sparkles);
  const composer = new EffectComposer(renderer);
  composer.setPixelRatio(S.dpr || 1);
  composer.setSize(W, H);
  composer.addPass(new RenderPass(scene, camera));
  const b = S.bloom || {};
  if (b.strength) composer.addPass(new UnrealBloomPass(new THREE.Vector2(W, H), b.strength, b.radius ?? 0.4, b.threshold ?? 0.85));
  composer.addPass(new OutputPass());
  composer.render();
  window.__ready = true;
}

build().catch((e) => { window.__error = String(e && e.stack || e); window.__ready = true; });
