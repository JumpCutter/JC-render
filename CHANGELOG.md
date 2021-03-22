# Change Log

## [0.0.2](https://github.com/JumpCutter/JC-render/compare/v0.0.1...v0.0.2) (03-22-2021)

### Bug Fixes
- __binaries__: added the necessary binaries for mac and linux to [render.spec](./render.spec)
- __package__: added a package and husky for future use.


## [0.0.1](https://github.com/JumpCutter/JC-render/compare/v0.0.0...v0.0.1) (02-13-2021)

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
