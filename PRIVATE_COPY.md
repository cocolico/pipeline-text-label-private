# Private development copy

This repository is intended to remain PRIVATE. It is a source snapshot of
`pipeline_text_label-20261016-172802/pipeline_text_label`, including local changes,
historical source backups, the Franka/Wuji demo and bundled MANO resources.
It is not an upstream release and does not imply rights to redistribute its code
or third-party models. Original model license files are retained.

## Snapshot boundaries

- Original source directory was left unchanged.
- Original `.git` history was not imported because it contains embedded credentials.
- Four embedded API credential occurrences were removed from copied source files.
- Supply `ARK_API_KEY`/`API_KEY` through the runtime environment. `.env.example`
  contains no credential; `.env` is ignored and is not automatically loaded.
- Virtual environments, bytecode, process logs, resource-monitor logs, accidental
  `--host`/`--port`/`--video` files, macOS metadata and host-specific symlinks
  were omitted. The machine-specific `bin/ffmpeg` link is not portable; use the
  installed `ffmpeg` executable.
- Bundled model ZIP/PKL files use Git LFS. Install Git LFS and run `git lfs pull`
  after cloning. Model filenames and directory layout are preserved.
- Demo videos/results outside the source folder were not imported.
- `MIGRATION_MANIFEST.json` lists copied source paths, hashes, omissions and
  credential-redaction counts. Hashes describe the sanitized initial snapshot,
  not future modifications or subsequently added management files.

## Current functionality

`api/api_seed_openai_easy_no_short_step/pipeline.py` is the original pipeline
entry point. It still needs upstream hand-screening data and robot adaptation
for a full Franka/Wuji workflow. `demos/franka_wuji/run_demo.py` is a separate
semantic-only demo; see its README for limitations. Legacy monitoring/remote
scripts may retain original-machine paths and should be configured before use.

Manage future development on feature branches. Do not change repository
visibility or commit API keys, runtime environments, or private dataset videos.
