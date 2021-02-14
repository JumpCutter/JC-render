# -*- mode: python ; coding: utf-8 -*-
import os

PACKAGE_DLL=os.getenv('PACKAGE_DLL', 'false') == 'true'

block_cipher = pyi_crypto.PyiBlockCipher(key='amMtcmVuZGVyZXI=')

binaries = []

if PACKAGE_DLL:
   binaries = [
      ('binaries/opentimelineio', '.'),
      ('binaries/opentimelineio_contrib', '.'),
      ('binaries/opentime.dll', '.'),
      ('binaries/opentimelineio.dll', '.'),
      ('binaries/redis', '.'),
   ]

a = Analysis(['render.py'],
             pathex=['/home/drei/repos/JC-renderer'],
             binaries=binaries,
             datas=[
               ('version.txt', '.'),
            ],
             hiddenimports=[
                'fractions',
                'xml.dom',
                'colorsys',
                'aaf2'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='render',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
