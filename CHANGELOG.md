# Change Log

## [0.0.7](https://github.com/JumpCutter/JC-render/compare/v0.0.6...v0.0.7) (2021-11-30)


## Bug Fixes
- __ascii__ : stderr from WHAT IS THIS now decoded with utf8 so decode doesn't throw with
non ascii characters in video file name

## [0.0.6](https://github.com/JumpCutter/JC-render/compare/v0.0.5...v0.0.6) (2021-07-23)

## Bug Fixes
- __vegasedl__: added to `supported_transfers` so that it exports instead of render


## [0.0.5](https://github.com/JumpCutter/JC-render/compare/v0.0.4...v0.0.5) (2021-06-23)

## Bug Fixes
- __Timestamp__: moved from "http://tsa.starfieldtech.com" (GoDaddy) to the digicert Time Stamp server.


## [0.0.4](https://github.com/JumpCutter/JC-render/compare/v0.0.3...v0.0.4) (2021-06-15)

### Changes
- __Vegas EDL generator__: Added a new export format. Exporting to Sony Vegas flavour of EDL. `vcodec` value for this format is `vegasedl`.


## [0.0.3](https://github.com/JumpCutter/JC-render/compare/v0.0.2...v0.0.3) (2021-04-01)

### Bug Fixes
- __Pixel Aspect Ratio__: added pixel aspect ratio for xml


## [0.0.2](https://github.com/JumpCutter/JC-render/compare/v0.0.1...v0.0.2) (2021-03-22)

### Bug Fixes
- __binaries__: added the necessary binaries for mac and linux to [render.spec](./render.spec)
- __package__: added a package and husky for future use.


## [0.0.1](https://github.com/JumpCutter/JC-render/compare/v0.0.0...v0.0.1) (2021-02-13)

### Changes
- __open source__: This is the real deal now
    - __updated readme__: README makes some sense now
    - __CHANGELOG__: added for devs
- __ci yaml__: multiple updates to make the ci production ready
- __example__: Added the examples submodule

### Performance Improvements
- __dry__: the chunking is dry now

### General Improvements
- __release sed__: the release can read our new CHANGELOG
- __version__: the release checks the CHANGELOG version

### Bug Fixes
- __windows signing__: signtool fixes most of the windows issues
- __chunking__: reimplemented chunks wit fix to edge case

### BREAKING CHANGES
- __ci yaml__: this only runs master
- __open source__:
    - __version__: the version was reset to 0.0.1

### DEPRECATIONS
- **none**: just here so devs see my formatting:)
