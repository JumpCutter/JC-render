import decimal
import os

def write_to_file(timeline, file_name):
    clips = transform_clips(timeline.each_clip())

    # Assuming same media ref for all clips
    videoFileName = os.path.abspath(next(timeline.each_clip()).media_reference.target_url)
    edl(file_name, clips, videoFileName)


def getRealTime(rationalTime):
    # make a normal number from otio RationalTime
    return rationalTime.value / rationalTime.rate

def getRealRange(trimmed_range):
    # returns starty and length
    return getRealTime(trimmed_range.start_time), getRealTime(trimmed_range.duration)

def transform_clips(clips):
    out_clips = list()
    i = 1
    for clip in clips:
        begin, length = getRealRange(clip.trimmed_range())
        if i != 1: # filling in the gaps
            out_clips.append(list())
            out_clips[-1].append(i) # ID
            i+=1
            between_begin = float(out_clips[-2][1]) + float(out_clips[-2][2])
            between_length = float(getMillis(begin)) - between_begin
            out_clips[-1].append(
                    decimal.Decimal(between_begin).quantize(decimal.Decimal('0.0001'))
                    )
            out_clips[-1].append(
                    decimal.Decimal(between_length).quantize(decimal.Decimal('0.0001'))
                    )

        out_clips.append(list())
        out_clips[-1].append(i) # ID
        i+=1
        out_clips[-1].append(
                getMillis(begin)
                )
        out_clips[-1].append(
                getMillis(length)
                )

    return out_clips


# do I need to do this?
def getMillis(time):
    time = (decimal.Decimal(time) * decimal.Decimal(1000))
    time = time.quantize(decimal.Decimal('.0001'))
    return str(time)


def edl(outfile, clips, videoFileName):
    f = open(outfile, 'w')
    f.write('"ID";          "Track";       "StartTime";    "Length";'
            '"PlayRate"    "Locked";      "Normalized";   "StretchMethod";'
            '"Looped";      "OnRuler";     "MediaType";    "FileName";'
            '"Stream";      "StreamStart"; "StreamLength"; "FadeTimeIn";'
            '"FadeTimeOut"; "SustainGain"; "CurveIn";      "GainIn";'
            '"CurveOut";    "GainOut";     "Layer";        "Color";'
            '"CurveInR";    "CurveOutR";   "PlayPitch";    "LockPitch"\n')
    for i, mtype in enumerate(["AUDIO", "VIDEO"]):
        for j, clip in enumerate(clips) :
            # Here are some default values. I Wish I knew what most of this does
            f.write(
                #| ID             |       Track |    StartTime | Length           |
                f'{clip[0]+(i *j)};          {i};     {clip[1]};         {clip[2]};' +

                #| PlayRate       |      Locked |   Normalized | StretchMethod    |
                f'        1.000000;        FALSE;         FALSE;                 0;' +

                #| Looped         |     OnRuler |    MediaType | FileName         |
                f'            TRUE;        FALSE;       {mtype}; "{videoFileName}";' +

                #| Stream         | StreamStart | StreamLength | FadeTimeIn       |
                f'               0;    {clip[1]};     {clip[2]};            0.0000;' +

                #| FadeTimeOut    | SustainGain |      CurveIn | GainIn           |
                f'          0.0000;     1.000000;             4;          0.000000;' +

                #| CurveOut       |     GainOut |        Layer | Color            |
                f'               4;     0.000000;             0;                -1;' +

                #| CurveInR       |   CurveiiOutR |    PlayPitch | LockPitch        |
                f'               4;            4;      0.000000;            FALSE\n'
                )
    f.close()
