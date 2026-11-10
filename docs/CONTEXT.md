# TurnSketch Studio: konteks proyek untuk sesi berikutnya

Dokumen ini merangkum apa yang sudah dibangun, apa yang sudah terbukti jalan,
apa yang belum, dan langkah selanjutnya. Tempel isinya (atau minta sesi baru
membaca file ini) sebelum melanjutkan pekerjaan.

- Repo: `virtuci/sketch-rotator`
- Branch: `claude/sketch-3d-rotation-tool-sgxh19` (sudah di-push, belum ada pull request)
- Detail teknis lengkap, struktur folder, dan referensi API: `README.md`

## 1. Tujuan

Aplikasi web self-hosted untuk studio anime. Artist menempelkan satu sketsa 2D
atau keyframe cel-shaded, memutarnya di 3D (yaw, pitch, roll, focal length
15–200mm dengan dolly-zoom), lalu mendapat layer 2D bersih dari sudut baru yang
bisa langsung di-paste kembali ke Clip Studio Paint, Photoshop, atau Krita.

Dua mode input:

- **`lineart`**: garis hitam 2–4px di latar putih atau transparan, tanpa shading.
- **`cel_chroma`**: karakter berwarna cel-shading 2-tone dengan outline hitam di
  atas green screen `#229919`. Hasil akhir dikomposit kembali ke `#229919` murni.

Kasus sulit yang harus ditangani: kacamata (harus ada parallax dengan pupil),
foreshortening ekstrem (tangan menunjuk kamera, memegang nampan atau cangkir),
karakter duduk di meja (pisahkan karakter dari furnitur), dan badge seragam
seperti logo OSIS (harus di-warp secara geometris, tidak digambar ulang).

## 2. Arsitektur

- **Frontend** (`frontend/`): Next.js App Router, TypeScript, Tailwind, Framer
  Motion, Three.js lewat `@react-three/fiber` dan `@react-three/drei`.
- **Backend** (`backend/`): gateway FastAPI (REST + WebSocket) dan worker GPU
  Celery, dengan Redis sebagai broker. Untuk dev ada antrean inline (thread
  pool di dalam gateway) sehingga tidak butuh Redis maupun GPU.
- **Deploy**: `docker-compose.yml` berisi `frontend`, `gateway`,
  `gpu-worker-N` (satu container per kartu GPU), `redis`, dan `caddy` (HTTPS).

Alur data:

1. `POST /api/v1/session/init`: buang background, bangun proxy 3D `.glb`,
   kembalikan `session_id`, URL tekstur, URL mesh, daftar layer, kamera asal.
2. Browser memutar proxy secara interaktif. Tidak ada beban ke server.
3. **Apply rotation**: browser merender 4 buffer (`projected_rgba`,
   `viewport_depth` 16-bit, `viewport_normal`, `disocclusion_mask`) lalu
   mengirimnya ke `POST /api/v1/session/{id}/refine`.
4. Progres dialirkan lewat `WS /api/v1/jobs/{job_id}/ws`. Hasilnya layer RGBA
   plus versi flat (di atas putih atau `#229919`).

## 3. Keputusan desain yang perlu diketahui

- **Konvensi dunia**: bidang gambar di z = 0, tinggi 2 unit, +z ke arah
  penonton. Kamera asal (sudut pandang saat sketsa digambar) ada di +z pada
  jarak di mana bidang itu tepat memenuhi frame. Focal asal default 85mm.
- **Dolly-zoom**: `distance = 1 / tan(fovY / 2)`, `fovY = 2·atan(12 / focal)`.
  Pada focal asal dan rotasi nol, preview identik dengan gambar asli.
- **Urutan rotasi**: Euler `ZXY`, supaya roll selalu di bidang gambar.
- **Deteksi area tak terlihat** (shader `ProjectiveDisocclusionMaterial`):
  `N·V ≤ 0.2` terhadap kamera asal, atau tertutup menurut shadow-map dari
  kamera asal, atau back face. Area ini tidak menampilkan gambar yang
  "tembus"; diisi dasar netral untuk refiner.
- **Capture**: satu render ke 4 attachment pada resolusi 2×, tanpa blending
  dan tanpa MSAA, lalu di-resolve di CPU (rata-rata warna, depth terdekat,
  normal dinormalisasi ulang, coverage masuk alpha). Depth dikemas 16-bit ke
  kanal R dan G.
- **Peta denoise** (untuk differential diffusion): permukaan yang terlihat
  jelas 0.05 (cel) atau 0.12 (lineart), naik sampai 0.35 di sudut miring,
  0.85 di area tak terlihat. Ini yang melindungi badge dan batas cel.
- **Layer**: `plane` (kacamata) adalah cangkang tipis yang melayang di depan
  permukaan; wajah tetap ada di bawahnya. `inflate` (meja, kursi, prop)
  dipotong dari layer dasar, punya volume sendiri, dan bisa dikunci agar
  tidak ikut berputar.
- **Matting chroma**: alpha tepi dihitung secara tertutup dari campuran
  `p = a·F + (1−a)·K`, bukan dengan erosi, sehingga piksel terluar outline
  hitam tetap utuh dan tidak ada sisa hijau. Default mengasumsikan key
  persis `#229919` dan anti-aliasing 1px (`tolerance=8`, `band_px=1`).
- **Siluet lineart**: flood fill dari tepi kanvas, dengan penutupan celah
  garis dan penutupan otomatis di tepi kanvas untuk gambar bust yang
  terpotong.

## 4. Status: apa yang sudah terbukti jalan

Semua ini jalan tanpa GPU dan tanpa bobot model:

- 26 test backend dan 13 test frontend lulus; frontend berhasil di-build.
- Aplikasi dijalankan end-to-end di Chromium headless untuk kedua mode:
  paste, deteksi mode otomatis, rotasi, ganti focal, apply, copy ke clipboard.
- Layer kacamata dari mask yang dicat menghasilkan parallax yang terlihat.
- Jalur Celery + Redis diuji dengan Redis dan worker sungguhan.

Catatan: semua pengujian memakai gambar sintetis buatan sendiri, belum aset
produksi.

## 5. Status: apa yang BELUM nyata

- **Tidak ada model generatif yang aktif.** Keempat hook di bawah mati secara
  default dan belum pernah dijalankan dengan bobot asli:

  | Model | Fungsi | Saklar | File |
  | --- | --- | --- | --- |
  | SDXL + ControlNet + IP-Adapter | Menggambar ulang area tak terlihat | `TS_REFINER=diffusion` | `backend/app/models/refiner.py` |
  | Hunyuan3D-2 | Mesh 3D sungguhan | `TS_RECONSTRUCTOR=hunyuan3d` | `backend/app/models/hooks.py` |
  | Depth-Anything-V2 | Kedalaman untuk proxy | `TS_DEPTH_ESTIMATOR=depth_anything` | `backend/app/models/hooks.py` |
  | SAM-2 | Pemisahan layer otomatis | `TS_SEGMENTER=sam2` | `backend/app/models/hooks.py` |

- **Proxy 3D** saat ini adalah siluet yang digembungkan (kepala jadi kubah,
  lengan jadi tabung), bukan rekonstruksi. `canonical_anime_head` hanya
  memilih preset gembung yang lebih bulat; belum ada base mesh kepala yang
  di-fit dengan landmark.
- **Refine** saat ini tidak menggambar apa pun. Piksel terlihat adalah warp
  geometris; area tak terlihat diisi kertas kosong plus garis siluet
  (lineart) atau warna lokal yang diblur (cel).
- **Pemotongan diskontinuitas depth** (mencegah lengan menyatu dengan dada)
  sudah ada dan teruji, tapi baru aktif jika model depth dinyalakan.
- **Pemisahan layer** masih lewat mask yang dicat manual dan diunggah dari
  panel samping. Prompt SAM-2 belum tersambung ke UI.
- **Image Docker belum pernah di-build**; file compose hanya lulus
  `docker compose config`.
- **Belum dibuat sama sekali**: pembersihan sesi di disk, autentikasi, dan
  plugin bridge DCC (Photoshop UXP, Krita, Blender).

## 6. Yang harus disiapkan pemilik proyek

1. **Server GPU**: driver NVIDIA, Docker, NVIDIA Container Toolkit. Perkiraan
   (belum diukur) minimal 16 GB VRAM per kartu, 24 GB lebih aman. Disk
   puluhan GB untuk bobot. Akses ke Hugging Face saat pertama jalan.
2. **Jaringan**: domain internal (`STUDIO_DOMAIN` di `.env`) dan HTTPS.
   Clipboard write hanya jalan di HTTPS; CA internal Caddy harus dipercaya
   di komputer artist, atau pakai sertifikat dari CA studio.
3. **Model**: tentukan checkpoint SDXL anime (default `animagine-xl-3.1`),
   sediakan ControlNet normal untuk SDXL (`TS_CONTROLNET_NORMAL` sengaja
   kosong), pasang Hunyuan3D-2 dan SAM-2 dari repo masing-masing, dan cek
   lisensi untuk pemakaian komersial.
4. **Aset uji produksi**: lineart dan keyframe `#229919` asli, termasuk
   kasus kacamata, tangan menunjuk kamera, memegang nampan atau cangkir,
   duduk di meja, dan badge OSIS. Untuk kacamata dan meja, siapkan mask yang
   dicat manual berukuran sama dengan gambar.

## 7. Langkah selanjutnya (urutan yang disarankan)

1. Deploy tanpa model (`cp .env.example .env`, lalu
   `docker compose up -d --build`) untuk membereskan build image, domain,
   HTTPS, dan clipboard.
2. Nyalakan `TS_REFINER=diffusion` dan uji dengan aset produksi. Ini yang
   paling menentukan kualitas dan paling butuh tuning (skala ControlNet,
   skala IP-Adapter, prompt, jumlah step).
3. Nyalakan Depth-Anything, lalu Hunyuan3D-2, lalu SAM-2, satu per satu.
   Mesh dari Hunyuan3D-2 perlu dicek orientasinya; `register_to_silhouette`
   mengasumsikan mesh menghadap +z dengan +y ke atas.
4. Setelah itu: UI prompt SAM-2, base mesh kepala anime dengan fitting
   landmark, pembersihan sesi, autentikasi, plugin bridge DCC.

## 8. Menjalankan secara lokal

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Buka http://localhost:3000 lalu paste gambar. Test:

```bash
cd backend && .venv/bin/python -m pytest
```

```bash
cd frontend && npm test && npm run typecheck
```
