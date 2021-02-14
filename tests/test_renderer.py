import pytest
from render import Render, ffprobe
# import _pytest.skipping
# import render
# import threading
import json


@pytest.mark.render
@pytest.mark.ffmpeg
def test_render():
    r = Render('output.json', './tmp',
               no_clean=True,
               silent_log=True,
               thread_alloc=2,
               # readable=True,
               verbose=False, vcodec='h264')
    # r.socketThread.daemon = True
    r.socketThread.start()
    r.render(False)
    r.keep_socket_open = False
    r.socketThread.join()

    print("ahh")


@pytest.mark.export
@pytest.mark.ffmpeg
@pytest.mark.xml
def test_export_xml():
    r = Render('output.json', './tmp',
               no_clean=True,
               # markers=True,
               # readable=True,
               verbose=False, vcodec='xml')
    r.export()


@pytest.mark.export
@pytest.mark.ffmpeg
@pytest.mark.fcpxml
def test_export_fcpxml():
    r = Render('output.json', './tmp',
               no_clean=True,
               # markers=True,
               # readable=True,
               verbose=False, vcodec='fcpxml')
    r.export()


@pytest.mark.export
@pytest.mark.ffmpeg
@pytest.mark.edl
def test_export_edl():
    r = Render('output.json', './tmp',
               no_clean=True,
               # readable=True,
               verbose=False, vcodec='edl')
    r.export()


@pytest.mark.export
@pytest.mark.ffmpeg
@pytest.mark.otio
def test_export_otio():
    r = Render('output.json', './tmp',
               no_clean=True,
               # readable=True,
               verbose=False, vcodec='otio')
    r.export()


@pytest.mark.export
@pytest.mark.ffmpeg
@pytest.mark.aaf
def test_export_aaf():
    r = Render('output.json', './tmp',
               no_clean=True,
               # readable=True,
               verbose=False, vcodec='aaf')
    r.export()


@pytest.mark.probe
@pytest.mark.ffmpeg
def test_probe():
    r = Render('output.json', './tmp', True, vcodec='h264')
    input_data = r.get_json(r.json_file)
    probe_res = ffprobe(input_data['layers'][0][0]['sourceFile'])
    print(json.dumps(probe_res, indent=2))


@pytest.mark.probe_audio
@pytest.mark.ffmpeg
def test_audio():
    r = Render('output.json', './tmp', True, vcodec='h264')
    input_data = r.get_json(r.json_file)
    layer = input_data['layers'][0][0]['sourceFile']
    probe_res = ffprobe(layer, select_streams='a')
    print(json.dumps(probe_res, indent=2))


def test_other():
    assert(True)
