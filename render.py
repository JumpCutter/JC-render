import re
import xml.etree.ElementTree as ET
import sys
import math
import json
import os
import subprocess
import argparse
import opentimelineio as otio
import signal
import vegasedl
from opentimelineio.schema import (
    Timeline,
    Track,
    Clip,
    ExternalReference,
    LinearTimeWarp,
    Gap,
    Marker,
    TrackKind
)
from opentimelineio.opentime import TimeRange, RationalTime
from pathlib import Path
import threading

import ffmpeg
import random
import string

from logger import create_logger
# TMP_FOLDER = os.getenv('jumpcutter_temp_dir', 'temp')
# FFMPEG_PATH = os.getenv('FFMPEG_PATH', 'ffmpeg')
# SPLIT_SIZE = int(os.getenv('SPLIT_SIZE', '100'))
# USE_GPU = os.getenv('GPU', 'False') == 'True'
# VCODEC = os.getenv('VCODEC', 'h264')
# WORKER_COUNT = int(os.getenv('WORKER_COUNT', '2'))
TMP_FOLDER = 'tmp'
FFMPEG_PATH = 'ffmpeg'
FFPROBE_PATH = 'ffprobe'
SPLIT_SIZE = 100
USE_GPU = False
# VCODEC = 'hevc'
VCODEC = 'h264'
WORKER_COUNT = 2
# COMPAT_TOOL_SUBSET = 60 * 60  # 1 hr in seconds
COMPAT_TOOL_SUBSET = 1 * 60  # big sad
LOG_NO_INFO = os.getenv('LOG_NO_INFO', 'true') == 'false'
possible_rates = [23.98, 24, 25, 29.97, 30, 50, 59.94, 60, 120, 144, 240]


def get_real_rate(test_rate):
    """Because fps is estimated, this finds the more likely closest real frame rate."""
    return min(sorted(possible_rates), key=lambda x: abs(x-test_rate))


def ffprobe(file_name: str, *args, **kwargs):
    """This is here so that I don't have to import ffmpeg directly in my tests."""
    # return ffmpeg.probe(file_name, select_streams='a')
    return ffmpeg.probe('{}'.format(file_name), *args, **kwargs)


def proportional(x: int = None, k: int = None):
    """Yields y=kx where x is directy proportional to y.

    It is literally just to make things more readable.

    :param x: the origional progress (layer_id)
    :param k: the constant of variation or constant of proportionality.
    """
    assert(x > 1), "this function does not allow inverse proportionality"
    return x * k


class Render:
    frame_rate = 29.97
    """if not detect_framerate
    the set frame rate
    else the rate from the last ffprobe action"""
    detect_framerate = False
    """force operations to use a preset rate or detect with ffprobe"""
    avg_frame_rate: float = None
    """the average rate detected by the last ffprobe action"""
    nb_frames: int = None
    """the total number of frames detected by the last ffprobe action"""
    current_codec = None
    """the chosen codec"""
    total_time = 0
    """the leng detected by the last ffprobe action"""
    bit_rate: int = None
    """the bit rate detected by the last ffprobe action"""
    longest_time = 0
    """longest length of a video (necessary for layering)"""
    total_layers = 0
    """the count of video layers"""
    tmp_files = []
    """a running list of files to be cleaned after each operation"""
    sounded_speed = 1.3  # TODO: make this inifinate capable
    """the speed of souned sections"""
    silent_speed = 0.6  # TODO: make this inifinate capable
    """the speed of silent sections"""
    markers = False
    """output to markers rather than cuts"""
    resolution = None
    """the resolution detected by the last ffprobe action"""
    socket = None
    """the current logging socket"""
    thread_alloc = 0
    """the number of treads to allocate where 0=all"""
    pixel_aspect_ratio = 1.0
    """the calculated pixel aspect ratio"""

    def __init__(self,  # NOQA
                 json_file: str,
                 tmp_folder=TMP_FOLDER,
                 no_clean=False,
                 readable=False,
                 no_info=False,
                 silent_log=False,
                 detect_framerate=True,
                 verbose=False,
                 quiet=False,
                 vcodec=VCODEC,
                 markers=False,
                 use_gpu=USE_GPU,
                 mac=False,
                 compat_tool_sub=COMPAT_TOOL_SUBSET,
                 thread_alloc=0,
                 ffmpeg_path=FFMPEG_PATH,
                 ffprobe_path=FFPROBE_PATH):
        """Create a Render class with all the inputs.
        To run call:
            socketThread.start()
            render()
            keep_socket_open = False
            socketThread.join()

        :param json_file: the path to the file containing all the edits
        :param tmp_folder: the location to store temp files
        :param no_clean: set True if you want to leave temp files after completioin
        :param readable: set true to output readable ffmpeg logs
        :param no_info: a dev param to skip info level logs in the stdout
        :param silent_log: output nothing to stdout nor stderr
        :param detect_framerate: let ffprobe detect the framerate
        :param verbose: pipe ffmpeg stdout to the stdout
        :param quiet: set the ffmpeg log level to error
        :param vcodec: the video codec to output
        :param markers: export to markers rather than cuts
        :param use_gpu: force the nvenc codec onto the vcodec
        :paran mac: force the videotoolbox codec onto the vcodec
        :param compat_tool_sub: a compatability measure to overcome the max shell length while running ffmpeg
        :param thread_alloc: force ffmpeg to use only n threads where n=0 is all
        :param ffmpeg_path: the path to ffmpeg
        :param ffprobe_path: the path to ffprobe
        """
        self.json_file = json_file
        self.tmp_folder = tmp_folder
        if not os.path.isdir(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        self.clean_tmp = not no_clean
        pid = os.getpid()
        self.keep_socket_open = True

        def process():
            while self.socket is None:
                pass
            with self.socket:
                while self.keep_socket_open:
                    try:
                        data = self.socket.recv(1024)
                        if not data:
                            break
                        if data.decode("utf-8") == 'exit':
                            break
                    except OSError:
                        pass
                return os.kill(pid, signal.SIGKILL)

        self.socketThread = threading.Thread(target=process)
        self.socketThread.daemon = True

        def socket_callback(socket):
            self.socket = socket

        self.logger = create_logger("render", tmp_folder, no_info, silent_log, socket_callback)
        self.verbosity = ('info', 'quiet')[quiet]
        self.detect_framerate = detect_framerate
        self.pipe_ffmpeg = verbose
        self.readable = readable
        self.vcodec = vcodec
        self.markers = markers
        self.max_tool_time = compat_tool_sub
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.thread_alloc = thread_alloc
        if use_gpu:
            self.vcodec = self.vcodec + '_nvenc'
        if mac:
            self.vcodec = self.vcodec + '_videotoolbox'

    def clean_tmp_files(self, tmp_files=None):
        """clear the tmp files created during this run"""
        if tmp_files is None:
            tmp_files = self.tmp_files
        self.logger.debug("clean_tmp_files -\n{}".format(tmp_files))
        for tmp_file in tmp_files:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    def get_json(self, json_file):
        """pretty simple json reader"""
        try:
            self.logger.debug("opening json - " + json_file)
            assert(os.path.exists(json_file)), 'json file not found'
            with open(json_file) as outfile:
                jc_edits = json.load(outfile, parse_constant=True)
                self.logger.debug(json.dumps(jc_edits, indent=2))
            return jc_edits
        except Exception as e:
            self.logger.exception(str(e))
            raise e

    def _handle_video_detection(self, probe_streams):
        """Because codecs store metadata differently we need many solutions
        frame_rate avg vs real are often incorrect."""
        if not self.resolution and "height" in probe_streams["streams"][0]:
            self.resolution = "{}x{}".format(
                probe_streams['streams'][0]["width"],
                probe_streams['streams'][0]["height"]
            )
        # avg_frame_rate = str(probe_streams["streams"][0]["avg_frame_rate"]).split("/")
        avg_frame_rate = str(probe_streams["streams"][0]["avg_frame_rate"]).split("/")
        self.avg_frame_rate = float(avg_frame_rate[0]) / float(avg_frame_rate[1])
        self.nb_frames = probe_streams["streams"][0].get("nb_frames")
        if self.vcodec != "aaf" or not self.frame_rate:
            r_frame_rate = probe_streams["streams"][0].get("r_frame_rate")
            if r_frame_rate:
                r_frame_rate = r_frame_rate.split("/")
                self.r_frame_rate = float(r_frame_rate[0]) / float(r_frame_rate[1])
            if self.avg_frame_rate:
                self.nb_frames = self.nb_frames or self.avg_frame_rate * self.total_time / 1000
            if self.detect_framerate:
                self.frame_rate = get_real_rate(self.r_frame_rate or self.avg_frame_rate)
        if "tags" in probe_streams["streams"][0]:
            if "NUMBER_OF_FRAMES" in probe_streams["streams"][0]["tags"]:
                self.nb_frames = probe_streams["streams"][0]["tags"]["NUMBER_OF_FRAMES"]
        try:
            self.nb_frames = int(self.nb_frames)
        except TypeError:
            self.logger.debug("cannot get nb_frames")

    def has_stream(self, file_name, stream='v'):  # NOQA
        """Run ffprobe and set all class variables"""
        hasit = False
        if self.ffprobe_path != 'ffprobe':
            self.logger.debug("direct ffrobe")
            assert(os.path.isfile(self.ffprobe_path)), 'invalid ffprobe path'
        try:
            try:
                probe_all = ffmpeg.probe('{}'.format(file_name), cmd=self.ffprobe_path)
            except Exception as e:
                self.logger.warning("WHAT IS THIS")  # FIXME: @JUL14N this one's for you buddy:)
                stderr = e.stderr.decode('ascii').replace('\n', '\n\t')
                self.logger.warning(f"ffprobe stderr:\n\t{stderr}")
            if stream == 'v':
                self.logger.debug("probe result -\n" + json.dumps(probe_all, indent=2))
            else:  # TODO: this is here to prove that we unnecessarily call has_stream twice
                self.logger.debug("TODO: called twice")
            try:
                if not self.resolution and 'height' in probe_all['streams'][0]:
                    self.resolution = "{}x{}".format(
                        probe_all['streams'][0]['width'],
                        probe_all['streams'][0]['height']
                    )
                self.current_codec = probe_all['streams'][0].get('codec_name')
                self.total_time = float(probe_all['format']['duration'])
                # self.bit_rate = float(probe_all['format']['bit_rate'])
                bit_rate = float(probe_all['format']['bit_rate'])
                if self.bit_rate or 0 < bit_rate:
                    self.bit_rate = bit_rate
            except KeyError:
                self.logger.warning("ffprobe key error")
            probe_streams = ffmpeg.probe(str(file_name), cmd=self.ffprobe_path,
                                         select_streams=stream)
            if stream == 'v':
                self._handle_video_detection(probe_streams)
                display_aspect_ratio = probe_all['streams'][0].get('display_aspect_ratio')
                sample_aspect_ratio = probe_all['streams'][0].get('sample_aspect_ratio')
                if display_aspect_ratio and sample_aspect_ratio:
                    dar = display_aspect_ratio.split(":")
                    sar = sample_aspect_ratio.split(":")
                    res = self.resolution.split("x")
                    x_pixel_ratio = float(res[1]) * float(dar[0]) / float(sar[0])
                    y_pixel_ratio = float(res[0]) * float(dar[1]) / float(sar[1])
                    self.pixel_aspect_ratio = x_pixel_ratio / y_pixel_ratio
            if probe_streams['streams']:
                hasit = True
        except Exception as e:
            self.logger.exception(str(e))
        return hasit

    def ffmpeg_out(self, current_layer: int, *args, override_nb_frames=False, **kwargs):  # NOQA
        """output with ffmpeg
        This takes in all normal ffmpeg output args.

        This is complex because we have to deal with default args and multiple output styles

        :param current_layer: the current index of the operation
        :param total_layers: the number of total layers to offset progress
        """
        total_layers = kwargs.pop('total_layers', self.total_layers)
        self.logger.debug(
            "ffmpeg_out - {}/{}".format(current_layer, total_layers)
        )
        if self.ffmpeg_path != 'ffmpeg':
            self.logger.debug("direct ffmpeg")
            assert(os.path.isfile(self.ffmpeg_path)), 'invalid ffmpeg path'

        kwargs['vcodec'] = kwargs.get('vcodec', self.vcodec)
        kwargs['threads'] = kwargs.get('threads', self.thread_alloc)
        if self.current_codec != kwargs['vcodec']:
            self.logger.debug("vcodec")
            if self.bit_rate is not None:
                kwargs['b'] = self.bit_rate
                kwargs['maxrate'] = self.bit_rate
                kwargs['bufsize'] = 2 * self.bit_rate
            kwargs['profile'] = 'high'
            kwargs['preset'] = 'slow'

        self.logger.debug("ffmpeg args - \n" + json.dumps(kwargs, indent=2))

        ffout = ffmpeg.output(*args, **kwargs).global_args(
            '-loglevel', self.verbosity, '-hide_banner'
        ).overwrite_output()
        if self.pipe_ffmpeg:
            ffmpeg.run(ffout, cmd=self.ffmpeg_path, overwrite_output=True)
        else:
            try:
                args = ffmpeg.compile(ffout, cmd=self.ffmpeg_path, overwrite_output=True)
                process = subprocess.Popen(
                    args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
                )

                for line in process.stdout:
                    if 'frame' in line and 'speed' in line:
                        groups = re.search(r'frame=\s*([0-9]*).*speed=\s*([0-9.]*)', line)  # noqa W605
                        frame_percent = int(groups.group(1)) / (self.total_time * self.frame_rate)
                        if not override_nb_frames and self.nb_frames is not None and self.nb_frames > 0:
                            frame_percent = int(groups.group(1)) / self.nb_frames
                        float_speed = 0.0
                        try:
                            float_speed = float(groups.group(2))
                        except ValueError:
                            pass

                        real_percent = frame_percent / total_layers + current_layer / total_layers
                        if self.readable:
                            sys.stdout.write("\033[K")  # clear previous line
                            self.logger.info(
                                '\r({}/{}) action progress is {:.3%} at a speed of {:}x'
                                .format(current_layer, total_layers, frame_percent, float_speed)
                            )
                        else:
                            self.logger.info('{:.3%}|{:.3f}x'.format(real_percent, float_speed))
                        self.logger.handlers[1].flush()
            except Exception as e:
                self.logger.exception(e)
                raise

    def cut_clips(self, edits, video=None, audio=None):  # NOQA
        self.logger.debug(self.total_time)
        edits.append({
            "start": edits[-1]["end"],
            "end": self.total_time
        })
        self.total_time = 0
        video_count = 0
        audio_count = 0
        video_split = None
        audio_split = None
        self.logger.debug("cut_clips -\n\tcut video - {}\n\tcut audio -{}".format(video, audio))
        if video is not None:
            video_split = video.filter_multi_output('split')
            video_count += 1
        if audio is not None:
            audio_split = audio.filter_multi_output('asplit')
            audio_count += 1
        try:
            cuts = []
            edit_sounded = self.sounded_speed is not None
            edit_silent = self.silent_speed is not None
            for i, edit in enumerate(edits):
                sounded_index = i
                silent_index = i
                if i >= len(edits) - 1:
                    break
                self.logger.debug("DSF: " + str(i))
                if edit_sounded and edit_silent:
                    sounded_index = i * 2
                    silent_index = sounded_index + 1

                if edit_sounded:
                    self.total_time += (edit['end'] - edit['start']) / self.sounded_speed
                    if video_split is not None:
                        video_cut = video_split[sounded_index].trim(
                            start=edit['start'], end=edit['end']
                        ).setpts(
                            'PTS-STARTPTS'
                        ).setpts(
                            "{}*PTS".format(1 / self.sounded_speed)
                        )
                        cuts.append(video_cut)
                    if audio_split is not None:
                        audio_cut = audio_split[sounded_index].filter(
                            'atrim', start=edit['start'], end=edit['end']
                        ).filter(
                            'asetpts', expr='PTS-STARTPTS'
                        ).filter(
                            'atempo', self.sounded_speed
                        )
                        cuts.append(audio_cut)
                if edit_silent:
                    self.total_time += (edits[i + 1]['start'] - edit['start']) / self.silent_speed
                    if video_split is not None:
                        video_cut_between = video_split[silent_index].trim(
                            start=edit['end'], end=edits[i + 1]['start']
                        ).setpts(
                            'PTS-STARTPTS'
                        ).setpts(
                            "{}*PTS".format(1 / self.silent_speed)
                        )
                        cuts.append(video_cut_between)
                    if audio_split is not None:
                        audio_cut_between = audio_split[silent_index].filter(
                            'atrim', start=edit['end'], end=edits[i + 1]['start']
                        ).filter(
                            'asetpts', expr='PTS-STARTPTS'
                        ).filter(
                            'atempo', self.silent_speed
                        )
                        cuts.append(audio_cut_between)
            edited = ffmpeg.concat(
                *cuts,
                v=video_count,
                a=audio_count
            )
            return edited
        except Exception as e:
            self.logger.exception(e)
            raise e

    def chunk_operation(
        self,
        cut_tuple,
        layer_id: int,
        appx_cut_count: float,
        cur_min: int,
        hr_op: int,
        adjusted_total: int,
        list_of_cuts=[]
    ):
        """Chunk the cut operations into smaller time-based sections
        :param cut_tuple: (per_minute_cuts, video, audio)
        :param layer_id: the current layer
        :param appx_cut_count: the guessed ammount of cuts necessary
        :param cur_min: the current minute index of the chunk
        :param hr_op: i dont remember
        :param list_of_cuts: the running total of cuts for ffmpeg to concat
        adjusted_total: the count of cuts adjusted for the number of streams
        """
        edits = [self.cut_clips(*cut_tuple)]
        adjusted_current = layer_id * appx_cut_count * 2 + cur_min - 1
        cut_name = "{}_{}_{}{}".format(hr_op[0], layer_id, cur_min, hr_op[1])
        cut_input = ffmpeg.input(cut_name)
        list_of_cuts.append(cut_input['v'])
        list_of_cuts.append(cut_input['a'])
        self.logger.debug("jumpcut - {}".format(cut_name))
        self.ffmpeg_out(adjusted_current, *edits, cut_name,
                        total_layers=adjusted_total)
        return adjusted_current

    def apply_operation(self, layer, layer_id, fps, resolution=None, speedrun=False):  # NOQA
        """create the ffmpeg options"""
        file_to_op = '{}'.format(layer['sourceFile'])
        opType = "none"
        try:
            if 'speed' in layer:
                layer_speed = layer['speed']
                # assert('sounded' in layer_speed), "speed object must have a `souneded` parameter"
                # assert('silent' in layer_speed), "speed object must have a `silent` parameter"
                self.sounded_speed = layer_speed.get('sounded', self.sounded_speed)
                self.silent_speed = layer_speed.get('silent', self.silent_speed)
                if self.sounded_speed == self.silent_speed:
                    # assert(self.sounded_speed is not None), "Malformatted speed is impossible."
                    if self.sounded_speed is None:
                        self.sounded_speed = 1
                        self.logger.warning("Malformatted speed is impossible.")
                    self.logger.warning(
                        "An equal sounded and silent speed is likely a waste of resources."
                    )
            else:
                self.sounded_speed = 1
                self.silent_speed = None
            assert(os.path.exists(file_to_op)), "cannot find operation file"
            opType = layer['opType']
            self.logger.debug("apply_operation - {} - {}".format(opType, file_to_op))
        except Exception as e:
            self.logger.exception(e)
            raise e
        source_file = ffmpeg.input(file_to_op)
        # op_format = ('.mkv', '.ts')[speedrun]
        # mp4 mkv mov avi mp3
        # op_format = self.vcodec
        # op_format = '.mp4'
        op_format = Path(self.outFile).suffix
        output_location = self.tmp_folder + '/test_' + str(layer_id) + op_format
        edits = []
        audio = None
        video = None
        total_layers = self.total_layers

        try:
            if self.has_stream(layer['sourceFile']):
                resolution = resolution or self.resolution
                video = source_file.video
                assert('x' in resolution), 'resolution should be a height and width separated by \'x\''
                dimensions = resolution.split('x')
                assert(len(dimensions) == 2), 'resolution should have 2 dimensions'
                width = dimensions[0]
                height = dimensions[1]
                video = video.filter(
                    'fps', fps=str(fps)
                ).filter(
                    'scale', width='min(' + width + ',iw)', height='min(' + height + ',ih)',
                    force_original_aspect_ratio='decrease'
                ).filter(
                    'pad', width=width, height=height, x='(ow-iw)/2', y='(oh-ih)/2'
                )
                if opType != 'jumpcutter':
                    video = video.setpts("{}*PTS".format(1 / (self.sounded_speed or self.silent_speed)))
                edits.append(video)
            if self.has_stream(layer['sourceFile'], 'a'):
                audio = source_file.audio
                if opType != 'jumpcutter':
                    audio = audio.filter("atempo", self.sounded_speed)
                edits.append(audio)
        except Exception as e:
            self.logger.exception(e)
            raise e
        try:

            if opType == 'jumpcutter':
                try:
                    assert(video is not None and audio is not None), "jumpcutter needs video and audio"
                except Exception as e:
                    self.logger.error(e)
                all_cuts = layer['cuts']
                cur_min = 1
                per_minute_cuts = []
                hr_op = output_location.split(str(layer_id))
                appx_cut_count = self.total_time / self.max_tool_time
                appx_cut_count = (appx_cut_count, 1)[appx_cut_count < 1]
                adjusted_total = appx_cut_count * (self.total_layers) * 2
                list_of_cuts = []
                for cut in all_cuts:
                    per_minute_cuts.append(cut)
                    if cut['start'] > self.max_tool_time * cur_min:
                        adjusted_current = self.chunk_operation(
                            (per_minute_cuts, video, audio),
                            layer_id,
                            appx_cut_count,
                            cur_min,
                            hr_op,
                            adjusted_total,
                            list_of_cuts,
                        )
                        cur_min += 1
                        per_minute_cuts = []
                # trimstart first ... last trimend
                if len(per_minute_cuts) > 0:
                    adjusted_current = self.chunk_operation(
                        (per_minute_cuts, video, audio),
                        layer_id,
                        appx_cut_count,
                        cur_min,
                        hr_op,
                        adjusted_total,
                        list_of_cuts,
                    )
                else:
                    adjusted_current = layer_id * appx_cut_count * 2 + cur_min

                edits = [ffmpeg.concat(*list_of_cuts, v=1, a=1)]

                layer_id = adjusted_current + 1
                total_layers = adjusted_total
        except Exception as e:
            self.logger.exception(e)
            raise
        try:
            opped_layer = {
                **layer,
                'sourceFile': output_location
            }
            self.logger.debug("operation finilize -\n{}".format(json.dumps(opped_layer, indent=2)))
            self.ffmpeg_out(layer_id, *edits, output_location, total_layers=total_layers)
            return opped_layer
        except Exception as e:
            self.logger.exception(e)
            raise

    def offset_layer(self, layer_data, i, resolution=None):
        resolution = resolution or self.resolution
        try:
            assert('sourceFile' in layer_data), "missing sourceFile in layer_data"
            assert('timelineStart' in layer_data), "missing timelineStart in layer_data"
        except Exception as e:
            self.logger.exception(e)
            raise
        source = layer_data['sourceFile']
        offset = layer_data['timelineStart']
        offset_seconds = float(offset) / 1000
        self.logger.debug("offset_layer - {}".format(offset_seconds))
        layer = ffmpeg.input(source)
        box = ffmpeg.input(
            'color=color=black@0.0:size={},format=rgba'.format(resolution),
            f='lavfi',
            t=str(offset_seconds + 2)
        )
        splits = {'video': None, 'audio': None}
        if self.has_stream(source):
            video = layer.video.setpts(
                'PTS-STARTPTS+{}/TB'.format(offset_seconds)
            )
            transparent_offset = ffmpeg.overlay(box, video, eof_action='repeat')
            video_split = transparent_offset.filter_multi_output('split')
            splits['video'] = [video_split[0], video_split[1]]
        if self.has_stream(source, 'a'):
            audio = layer.audio.filter(
                'adelay', delays='{}:all=1'.format(offset)
            )
            audio_split = audio.filter_multi_output('asplit')[0]
            splits['audio'] = audio_split
        self.total_time += offset / 1000
        return splits

    def overlay_layers(self, parent_split, overlay_split):
        output_splits = {'video': None, 'audio': None}
        self.logger.debug("overlay_layers")

        video_count = 0
        if parent_split['video'] is not None:
            video_count += 1
            output_splits['video'] = [parent_split['video'][0], parent_split['video'][1]]
        if overlay_split['video'] is not None:
            video_count += 1
            output_splits['video'] = [overlay_split['video'][0], overlay_split['video'][1]]
        if video_count > 1:
            try:
                self.logger.debug("overlay video")
                top_overlay_0 = ffmpeg.overlay(
                    parent_split['video'][0], overlay_split['video'][0], eof_action='pass'
                )
                bottom_overlay_0 = ffmpeg.overlay(
                    parent_split['video'][1], overlay_split['video'][1], eof_action='repeat'
                )
                output_splits['video'] = [top_overlay_0, bottom_overlay_0]
            except Exception as e:
                self.logger.exception(e)
                raise

        audio_count = 0
        if parent_split['audio'] is not None:
            audio_count += 1
            output_splits['audio'] = parent_split['audio']
        if overlay_split['audio'] is not None:
            audio_count += 1
            output_splits['audio'] = overlay_split['audio']
        if audio_count > 1:
            try:
                self.logger.debug("overlay audio")
                combined_audio = ffmpeg.filter(
                    [parent_split['audio'], overlay_split['audio']],
                    'amix', inputs=2, duration='longest'
                )
                output_splits['audio'] = combined_audio
            except Exception as e:
                self.logger.exception(e)
                raise
        return output_splits

    def clip_export(self, sourceFile, external_reference, mob_id, start_frame, end_frame, time_scalar):
        width, height = self.resolution.split("x")
        clip = Clip(
            sourceFile,
            external_reference,
            TimeRange(
                RationalTime(start_frame, self.frame_rate),
                RationalTime((end_frame - start_frame) * time_scalar, self.frame_rate)
            ),
            metadata={
                "fcp_xml": {"media": {"video": {
                    "samplecharacteristics": {
                        "height": height,
                        "width": width
                    }
                }}},
                "cmx_3600": {"reel": "AX"},
                "AAF": {
                    "SourceID": mob_id
                }
            }
        )
        if time_scalar != 1:
            clip.effects.append(LinearTimeWarp(time_scalar=time_scalar))
        return clip


    def render(self, speedrun=False):  # NOQA
        """This is how we generate a file directly:
            Before the for loop we setup and read inputps
            The first for loop generates the ffmpeg options to
            The second for loop layers the videos
            Where we run ffmpeg_out, we start the real rendering
        """
        input_data = self.get_json(self.json_file)
        try:
            # assert("resolution" in input_data), "input needs to specify resolution"
            assert("frameRate" in input_data), "input needs to specify frameRate"
            assert("outFile" in input_data), "input needs to specify outFile"
            assert("layers" in input_data), "input needs to specify layers"
        except Exception as e:
            self.logger.exception(e)
            raise
        # TODO: switch all cases to use self.resolution
        if 'resolution' in input_data:
            self.resolution = input_data['resolution']
        # resolution = input_data['resolution']
        resolution = self.resolution
        self.frame_rate = input_data['frameRate']
        frame_rate = str(self.frame_rate)
        outFile = str(input_data['outFile'])
        # FIXME: classwide outfile does not exist in export
        self.outFile = outFile
        layers = input_data['layers']

        flat_layers = [item for sublist in layers for item in sublist]
        operation_data = []
        self.total_layers = len(flat_layers)
        self.total_layers += 1

        try:
            assert(self.total_layers > 1), 'cannot run operation with 0 layers'
        except Exception as e:
            self.logger.exception(e)
            raise

        for i, layer in enumerate(flat_layers):
            self.sounded_speed = 1
            self.silent_speed = None
            opped_layer = self.apply_operation(layer, i, frame_rate, resolution, speedrun)
            self.tmp_files.append(opped_layer['sourceFile'])
            operation_data.append(opped_layer)

        output = []
        parent = self.offset_layer(
            operation_data[-1], i, resolution
        )
        reversed_operations = reversed(operation_data)
        if len(operation_data) > 1:
            for i, op in enumerate(reversed_operations):
                if i > 0:
                    overlay = self.offset_layer(
                        op, i, resolution
                    )
                    overlayed = self.overlay_layers(
                        parent,
                        overlay
                    )
                    parent = overlayed
                if self.total_time > self.longest_time:
                    self.longest_time = self.total_time
            try:
                self.logger.debug("final overlay -\n{}\n{}".format(
                    parent['video'][1],
                    parent['video'][0]
                ))
                parent['video'][0] = ffmpeg.overlay(
                    parent['video'][1], parent['video'][0], eof_action='pass'
                )
            except Exception as e:
                self.logger.exception(e)
                raise
            self.total_time = self.longest_time

        try:
            self.logger.debug("creating final output")
            if parent['video'] and len(parent['video']) > 0:
                output_video = parent['video'][0].filter(
                    'fps', fps=str(frame_rate)
                )
                output.append(output_video)
            if parent['audio'] is not None:
                output.append(parent['audio'])

            self.ffmpeg_out(self.total_layers - 1, *output, outFile, override_nb_frames=True)

        except Exception as e:
            self.logger.exception(e)
            raise

        if self.clean_tmp:
            try:
                self.logger.debug("cleaning -\n{}".format(json.dumps(self.tmp_files)))
                self.clean_tmp_files(self.tmp_files)
            except Exception as e:
                self.logger.exception(e)
                raise
        self.logger.debug("COMPLETE\n")
        self.logger.info("")

    def export(self, name: str = None):  # NOQA
        """This is a nightmare so here's what's happening:
            Before the for loop we're doing setup
            There are 2 for loops to deal with layering
            The inner for loop sets up each layer then at its last try, it creates a Track
            The outer for loops gathers the Tracks and creates a Timeline
            The timeline can be exported to files
        Good luck bruddah.
        """
        input_data = self.get_json(self.json_file)
        try:
            assert("frameRate" in input_data), "input needs to specify frameRate"
            assert("outFile" in input_data), "input needs to specify outFile"
            assert("layers" in input_data), "input needs to specify layers"
        except Exception as e:
            self.logger.exception(e)
            raise

        name = ".".join(input_data["outFile"].split(".")[:-1])

        self.frame_rate = input_data['frameRate']
        layers = input_data['layers']

        try:
            assert(len(layers) > 0), 'cannot run operation with 0 layers'
        except Exception as e:
            self.logger.exception(e)
            raise

        tracks = []
        if self.vcodec == "edl":
            if len(layers) > 1:
                self.logger.warning("EDL does not support layering")
                self.logger.info("Each layer will be output to separate files")

        for i, layer in enumerate(reversed(layers)):
            audio_clips: [Clip] = []
            video_clips: [Clip] = []
            markers: [Marker] = []
            self.nb_frames = None
            for clip in layer:
                edits = clip.get("cuts")
                self.has_stream(clip["sourceFile"])
                if self.nb_frames is None:
                    self.logger.error("nb_frames cannot be None")
                    assert(self.nb_frames is not None), "nb_frames cannot be None"

                self.sounded_speed = 1
                self.silent_speed = None
                if "speed" in clip:
                    self.sounded_speed = clip["speed"].get("sounded", self.sounded_speed)
                    self.silent_speed = clip["speed"].get("silent", self.silent_speed)
                    if self.vcodec not in ["edl", "otio"]:
                        self.sounded_speed = 1
                        if self.silent_speed != 1:
                            self.silent_speed = None
                        self.logger.warning("Only the edl format supports custom speed")
                        self.logger.info("Cuts will be returned instead")

                edit_sounded = self.sounded_speed is not None
                edit_silent = self.silent_speed is not None

                if clip["opType"] != "jumpcutter":
                    edits = [{
                        "start": 0.0,
                        "end": self.total_time
                    }]
                else:
                    if clip.get("timelineStart", 0) != 0:
                        self.logger.warning("Jumpcutting and offsetting are not compatable")
                        self.logger.info("Your start time will be 0s")
                        clip["timelineStart"] = 0

                edits[0]["start"] = 0.0

                timelineStart = clip["timelineStart"]
                offset = 1
                if len(layers) > 1 and timelineStart > 0:
                    offset = int(timelineStart / 1000 * self.frame_rate)
                    gap = (
                        "timelineStart",
                        TimeRange(
                            RationalTime(0, self.frame_rate),
                            RationalTime(offset, self.frame_rate)
                        )
                    )
                    video_clips.append(Gap(*gap))
                    audio_clips.append(Gap(*gap))
                    self.nb_frames = self.nb_frames * offset

                if edits[-1]["start"] != edits[-1]["end"]:
                    edits.append({
                        "start": edits[-1]["end"],
                        "end": edits[-1]["end"],
                    })
                try:
                    mob_id = "".join(random.choices(string.digits, k=64))
                    width, height = self.resolution.split("x")
                    external_reference = ExternalReference(
                        clip["sourceFile"],
                        TimeRange(
                            RationalTime(0, self.frame_rate),
                            # RationalTime((end_frame - start_frame), self.frame_rate)
                            RationalTime(self.nb_frames / offset, self.frame_rate)
                        ),
                        {
                            "AAF": {"MobID": mob_id},
                            'streaming': {
                                'width': width,
                                'height': height
                            }
                        }
                    )
                    offset = 0
                    for j, edit in enumerate(edits):
                        if j >= len(edits) - 1:
                            if self.markers:
                                marked_clip = self.clip_export(
                                    clip["sourceFile"],
                                    external_reference,
                                    mob_id,
                                    0, self.nb_frames,
                                    1 / self.sounded_speed
                                )
                                if len(markers) > 1:
                                    marked_clip.markers.extend(markers)
                                video_clips.append(marked_clip)

                            break

                        if self.markers:
                            markers.append(Marker(
                                "jumpcut",
                                TimeRange(
                                    RationalTime(
                                        math.floor((edit["start"] + offset) * self.frame_rate),
                                        self.frame_rate
                                    ),
                                    RationalTime(
                                        math.floor((edit["end"] + offset) * self.frame_rate),
                                        self.frame_rate
                                    )
                                ),
                                "PURPLE"
                            ))
                        else:
                            if edit_sounded:
                                sounded_edit = (
                                    clip["sourceFile"],
                                    external_reference,
                                    mob_id,
                                    math.floor((edit["start"] + offset) * self.frame_rate),
                                    math.floor((edit["end"] + offset) * self.frame_rate),
                                    1 / self.sounded_speed
                                )
                                video_clip = self.clip_export(*sounded_edit)
                                audio_clip = self.clip_export(*sounded_edit)
                                video_clips.append(video_clip)
                                audio_clips.append(audio_clip)
                            if edit_silent:
                                silent_edit = (
                                    clip["sourceFile"],
                                    external_reference,
                                    mob_id,
                                    math.floor((edit["end"] + offset) * self.frame_rate),
                                    math.floor((edits[j + 1]["start"] + offset) * self.frame_rate),
                                    1 / self.silent_speed
                                )
                                video_clips.append(
                                    self.clip_export(*silent_edit)
                                )
                                audio_clips.append(
                                    self.clip_export(*silent_edit)
                                )

                except Exception as e:
                    self.logger.exception(e)
                    raise
            try:
                width, height = self.resolution.split("x")
                video_track = Track(
                    "v{}".format(int(i)),
                    video_clips,
                    # None,
                    TimeRange(
                        RationalTime(0, self.frame_rate),
                        RationalTime(self.nb_frames, self.frame_rate)
                    ),
                    TrackKind.Video,
                    metadata={
                        'streaming': {
                            'width': width,
                            'height': height
                        }
                    }
                )
                tracks.append(video_track)
                audio_track = Track(
                    "aa{}".format(int(i)),
                    audio_clips,
                    # None,
                    TimeRange(
                        RationalTime(0, self.frame_rate),
                        RationalTime(self.nb_frames, self.frame_rate)
                    ),
                    TrackKind.Audio,
                    metadata={
                        'linked_tracks': [video_track.name]
                    }
                )
                tracks.append(audio_track)
            except Exception as e:
                self.logger.exception(e)
                raise

        try:
            timeline_name = Path(input_data["outFile"]).stem
            width, height = self.resolution.split("x")
            if self.vcodec == "edl":
                for i in range(0, len(tracks) - 1, 2):
                    timeline = Timeline(
                        # "AX",
                        "{}{}.edl".format(timeline_name, int(i / 2)),
                        # [tracks[i - 1], tracks[i]],
                        [tracks[i], tracks[i + 1]],
                        RationalTime(0, self.frame_rate)
                    )
                    otio.adapters.write_to_file(
                        timeline,
                        "{}{}.edl".format(name, int(i / 2))
                    )
            elif self.vcodec == "vegasedl":
                for i in range(0, len(tracks) - 1, 2):
                    timeline = Timeline(
                        f"{timeline_name}{int(i/2)}.txt",
                        [tracks[i], tracks[i + 1]],
                        RationalTime(0, self.frame_rate)
                    )
                    vegasedl.write_to_file(
                            timeline,
                            f"{name}{int(i / 2)}.txt"
                            )
            else:
                file_name = "{}.{}".format(name, self.vcodec)
                timeline = Timeline(
                    "{}.{}".format(timeline_name, self.vcodec),
                    tracks,
                    RationalTime(0, self.frame_rate)
                )

                otio.adapters.write_to_file(timeline, file_name)
                if self.vcodec == "xml":
                    xml_tree = ET.parse(file_name)
                    for seq in xml_tree.iter('sequence'):
                        video = seq.find("media").find("video")
                        xml_height = ET.Element("height")
                        xml_height.text = height
                        xml_width = ET.Element("width")
                        xml_width.text = width
                        xml_pixel_aspect_ratio = ET.Element("pixelaspectratio")
                        xml_pixel_aspect_ratio.text = str(self.pixel_aspect_ratio)
                        xml_chars = ET.Element(
                            "samplecharacteristics",
                        )
                        xml_chars.append(
                            xml_height
                        )
                        xml_chars.append(
                            xml_width
                        )
                        xml_chars.append(
                            xml_pixel_aspect_ratio
                        )
                        #              )
                        xml_format = ET.Element(
                            "format"
                        )
                        xml_format.append(xml_chars)
                        video.append(xml_format)
                        xml_tree.write(file_name)

            self.logger.debug("COMPLETE\n")
            self.logger.info("")
        except Exception as e:
            self.logger.exception(e)
            raise


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Render the video with layers and cuts.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('json_file', help='the json file to read')
    parser.add_argument('tmp_path', nargs='?', default='./tmp',
                        help='where to store the tmp files')
    parser.add_argument('--no_clean', default=False, action='store_true',
                        help='do not clean the created tmp files')
    parser.add_argument('-v', '--version', action="version",
                        version=open(resource_path("version.txt"), "r").read(),
                        help='get the version')
    # output
    output_group = parser.add_argument_group('output')
    output_group.add_argument('--readable', default=False, action='store_true',
                              help='pipe ffmpeg to human readable format')
    output_group.add_argument('--verbose', default=False, action='store_true',
                              help='pipe ffmpeg to stdout')
    output_group.add_argument('-q', '--quiet', default=False, action='store_true',
                              help='set ffmpeg verbosity to error')
    output_group.add_argument('--no_info', default=False, action='store_true',
                              help='skip logging info/stdout')
    output_group.add_argument('--silent', default=False, action='store_true',
                              help='skip logging all levels to stdout')
    # performance
    performance_group = parser.add_argument_group('performance')
    performance_group.add_argument('--compat_tool_sub', default=COMPAT_TOOL_SUBSET,
                                   help='this is the maximum time (sec) to process at once')
    performance_group.add_argument('--thread_alloc', type=int, default=0,
                                   help="(experimental) choose the number of threads to allocate for ffmpeg")
    # codecs
    codecs_group = parser.add_argument_group(
        'codecs', "use vc to set any vcodec or output format like aaf, xml, or edl"
    )
    codecs_group.add_argument('-vc', '--vcodec', default='h264',
                              help='choose the specific code to output')
    supported_transfers = ["aaf", "edl", "fcpxml", "otio", "xml", "vegasedl"]
    codecs_group.add_argument('-t', '--transfer',
                              choices=supported_transfers,
                              help="transfer to another software")
    codecs_group.add_argument('-m', '--markers', action='store_true',
                              help="(experimental) use marks with transfer")
    codecs_group.add_argument('--mac', default=False, action='store_true',
                              help='use videotoolbox')
    codecs_group.add_argument('-s', '--speedrun', default=False, action='store_true',
                              help='quick run cuts')
    codecs_group.add_argument('-g', '--gpu', default=False, action='store_true',
                              help='the json file to read')
    # ffmpeg
    ffmpeg_group = parser.add_argument_group('ffmpeg')
    ffmpeg_group.add_argument('--ffmpeg', default='ffmpeg',
                              help='the path to ffmpeg')
    ffmpeg_group.add_argument('--ffprobe', default='ffprobe',
                              help='the path to ffprobe')
    ffmpeg_group.add_argument('--detect_framerate', default=True, action='store_false',
                              help='disable the (experimental) framerate detection')
    # parser.add_argument('-t', default='ffmpeg',
    #                     help='choose the specific code to output')
    args = parser.parse_args()
    if args.transfer:
        args.vcodec = args.transfer
    r = Render(
        args.json_file, args.tmp_path, args.no_clean, args.readable,
        args.no_info, args.silent, args.detect_framerate,
        args.verbose, args.quiet, args.vcodec, args.markers, args.gpu, args.mac,
        args.compat_tool_sub, args.thread_alloc, args.ffmpeg, args.ffprobe
    )

    if args.vcodec in supported_transfers:
        r.export()
    else:
        r.socketThread.start()
        r.render(args.speedrun)
        # r.keep_socket_open = False
        r.socketThread.join()
