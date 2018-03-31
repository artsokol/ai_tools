#!/usr/bin/python2.7
# coding: utf8
import wave
import sys
import os
import time
import math
import argparse
import base64
import json
import simplejson
import httplib2
import time
import sox
import numpy
import threading
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioTrainTest as aT

from pydub import AudioSegment
from auditok import ADSFactory, AudioEnergyValidator, StreamTokenizer, player_for, dataset

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import matplotlib.pyplot as plt




def slice(infile, outfilename, start_ms, end_ms):
    width = infile.getsampwidth() #Returns sample width in bytes
    rate = infile.getframerate() #Returns sampling frequency
    fpms = rate / 1000 # frames per ms
    length = (end_ms - start_ms) * fpms
    start_index = start_ms * fpms
    out = wave.open(outfilename, "w")
    out.setparams((infile.getnchannels(), width, rate, length, infile.getcomptype(), infile.getcompname()))
    infile.rewind()       #Rewind the file pointer to the beginning of the audio stream
    anchor = infile.tell()   #Return current file pointer position
    infile.setpos(anchor + start_index) #Set the file pointer to the specified position
    out.writeframes(infile.readframes(length)) #Write audio frames and make sure nframes is correct

def get_speech_service():
    credentials = GoogleCredentials.get_application_default().create_scoped(
        ['https://www.googleapis.com/auth/cloud-platform'])
    http = httplib2.Http()
    credentials.authorize(http)

    return discovery.build(
        'speech', 'v1beta1', http=http, discoveryServiceUrl=DISCOVERY_URL)

def get_translate_list(speech_file):
    with open(speech_file, 'rb') as speech:
        speech_content = base64.b64encode(speech.read())

    service = get_speech_service()
    service_request = service.speech().syncrecognize(
        body={
            'config': {
                'encoding': 'FLAC',
                'sampleRate': 16000,  # 16 khz
                'languageCode': 'ru-RU',  # a BCP-47 language tag
            },
            'audio': {
                'content': speech_content.decode('UTF-8')
                }
            })
    response = service_request.execute()
    list_of_results = response.get('results')
    trancript_list = []

    if list_of_results == None:

        return trancript_list

    for res in list_of_results:
        #get the first alternative only
        alternative = res.get('alternatives')[0]
        confidence = alternative.get('confidence')
        result_str = alternative.get('transcript')
        trancript_list.append((result_str,confidence))

    return trancript_list

def cut_wav_file_on_segments(segments, audio_file):
    current_part = 0
    parts_list = []
    for begin, end in segments:
        begin_orig = begin
        if begin > 2000:
            begin -= 2000
        else:
            begin = 0
        begin_ms = int(begin)

        end_ms = int(end)
        if current_part != len(segments)-1:
            end_ms -= 1000

        part_file_name = os.path.splitext(audio_file)[0] + "_" + str(current_part) + os.path.splitext(audio_file)[1]

        #print(begin_ms)
        #print(end_ms)
        slice(wave.open(audio_file, "r"), part_file_name, begin_ms, end_ms)
        current_part+=1
        parts_list.append((begin_orig, part_file_name))
    return parts_list


def find_music(audio_file):
    modelName = "pyAA/data/svmSM"

    [Fs, x] = aIO.readAudioFile(audio_file);
    duration = x.shape[0] / float(Fs)
    t1 = time.clock()
    flagsInd, classNames, acc, CMt = aS.mtFileClassification(audio_file, modelName, "svm", False, '')
    [Classifier, MEAN, STD, classNames, mtWin, mtStep, stWin, stStep, computeBEAT] = aT.loadSVModel(modelName)
    t2 = time.clock()
    perTime1 =  duration / (t2-t1)
    flags = [classNames[int(f)] for f in flagsInd]
    (segs, classes) = aS.flags2segs(flags, mtStep)

    i = 0 #len(classes)-1
    file_parts=[]

    cbn = sox.Combiner()
    if len(classes) > 1:
        for c in classes:
            if c == 'music':
                start = segs[i][0]
                if i != 0:
                    start-=0.5
                end = segs[i][1]
                if i != len(classes)-1:
                    end+=2.5

                file_parts.append((int(start*1000),int(end*1000)))
            i+=1

    return file_parts

def find_voice_segments(audio_file, music_time_list):
    segments = []
    formats = {1: numpy.int8, 2: numpy.int16, 4: numpy.int32}
    #[Fs_cr, x_cr] = aIO.readAudioFile(input_audio_audio_file)
    #[Fs_ce, x_ce] = aIO.readAudioFile(callee_audio_file)
    #segments = aS.silenceRemoval(x_cr, Fs_cr, 0.010, 0.010, smoothWindow=3,Weight=0.3,plot=False)
    #print(segments)
    #callee_segments = aS.silenceRemoval(x_ce, Fs_ce, 0.010, 0.010, smoothWindow=5,Weight=0.3,plot=False)
    #print(callee_segments)

    test_source = ADSFactory.ads(filename=audio_file, record=False)
    test_source.open()
    i = 0
    max_value = 0.0
    a = numpy.empty([], dtype=numpy.float64)
    b = numpy.empty([], dtype=numpy.float64)
    while True:
        frame = test_source.read()

        if frame is None:
            break

        signal = numpy.array(numpy.frombuffer(frame, dtype=formats[test_source.get_sample_width()]),
                               dtype=numpy.float64)
        energy = float(numpy.dot(signal, signal)) / len(signal)
        max_value = max(max_value, energy)
        i+=1
        b = numpy.append(b, [energy])

    #diff = max_value - numpy.mean(b)
    #print(10. * numpy.log10(0.3*diff))
    log_max = 10. * numpy.log10(max_value)
    log_mean = 10. * numpy.log10(numpy.mean(b))
    tmp = log_max - log_mean
    threshold = log_mean + 0.4*tmp
    #print(threshold)

    test_source.close()
    asource = ADSFactory.ads(filename=audio_file, record=False)
    validator = AudioEnergyValidator(sample_width=asource.get_sample_width(), energy_threshold=threshold)
    tokenizer = StreamTokenizer(validator=validator, min_length=300, max_length=99999999, max_continuous_silence=300)
    player = player_for(asource)

    asource.open()
    tokens = tokenizer.tokenize(asource)
    for i, t in enumerate(tokens):
        segment_begin = t[1]*10
        segment_end = t[2]*10

        if len(music_time_list) > 0:
            for item in music_time_list:
                # if segment end includes music begin
                if segment_end > item[0]:
                    #include segment before music
                    segments.append([segment_begin, item[0]])
                    #save stamps for incluing segment after music
                    segment_begin=item[1]
                    # remove music segment from list
                    # to not use it in further
                    music_time_list.remove(item)

        segments.append([segment_begin, segment_end])

    asource.close()
    return segments

def normalize_file(audio_file):
    normalized_prefix = "normalized"
    normalized_file_name = os.path.splitext(audio_file)[0] + "_" + normalized_prefix + os.path.splitext(audio_file)[1]
    os.system("sox --norm " + audio_file + " " + normalized_file_name) #
    #  create the output file.
    os.remove(audio_file)
    os.rename(normalized_file_name, audio_file)

def normalize_file2(audio_file):
    normalized_prefix = "normalized"
    normalized_file_name = os.path.splitext(audio_file)[0] + "_" + normalized_prefix + os.path.splitext(audio_file)[1]
    sound = AudioSegment.from_file(audio_file, "wav")
    change_in_dBFS = -30.0 - sound.dBFS
    normalized_sound = sound.apply_gain(change_in_dBFS)
    normalized_sound.export(normalized_file_name, format="wav")
    #  create the output file.
    os.remove(audio_file)
    os.rename(normalized_file_name, audio_file)

def normalize_parts(parts):
    if len(parts) > 0:
        for part in parts:
            normalize_file(part[1])


def denoize_parts(parts):
    if len(parts) > 0:
        noise_profile = "./speech.noise-profile"
        denoised_prefix = "denoised"
        i = 0
        for part in parts:
            denoised_part_file_name = os.path.splitext(part[1])[0] + "_" + denoised_prefix + os.path.splitext(part[1])[1]
            if i == 0:
                #  always skip the first item
                i+=1
                os.system ("cp " + part[1] + " " + denoised_part_file_name)
                continue

            os.system("sox -c 1 " + part[1] + " -n trim 0 0.5 noiseprof " + noise_profile)
            # create the output file.
            os.system("sox " + part[1] + " " + denoised_part_file_name + " noisered " + noise_profile + " 0.005")
            os.remove(part[1])
            os.remove(noise_profile)
            os.rename(denoised_part_file_name, part[1])

def compand_file(audio_file):
    companded_prefix = "companded"
    companded_file_name = os.path.splitext(audio_file)[0] + "_" + companded_prefix + os.path.splitext(audio_file)[1]
    os.system("sox " + audio_file + " " + companded_file_name + " compand 0.02,0.20 5:-60,-40,-10 -5 -90 0.1")#-v 0.99 #fade 0.1
    #os.system("sox " + part[1] + " " + companded_part_file_name + " remix - highpass 100 norm compand 0.05,0.2 6:-54,-90,-36,-36,-24,-24,0,-12 0 -90 0.1 vad -T 0.6 -p 0.2 -t 5 fade 0.1 reverse vad -T 0.6 -p 0.2 -t 5 fade 0.1 reverse norm -0.5")
    # create the output file.
    os.remove(audio_file)
    os.rename(companded_file_name, audio_file)

def compand_parts(parts):
    if len(parts) > 0:
        for part in parts:
            compand_file(part[1])

def wav2flac(audio_file):
    flac_audio_file = os.path.splitext(audio_file)[0] + ".flac"
    os.system("sox " + audio_file + " " + flac_audio_file)
    return flac_audio_file

def translator(position, record_item, final_list):
        audio = wav2flac(record_item[1])
        #print(str(position))
        final_list[position] = (record_item[0],get_translate_list(audio))
        os.remove(audio)

if __name__ == '__main__':

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
    os.environ["GCLOUD_PROJECT"] = ""
    os.environ["PYTHONIOENCODING"] = "UTF-8"

    DISCOVERY_URL = ('https://{api}.googleapis.com/$discovery/rest?'
                     'version={apiVersion}')

    out_data = {} 
    #TODO: just one stream as input
    input_audio_audio_file = sys.arsgv[1]
    #callee_audio_file = sys.argv[2]

    #Create tmp copy
    input_audio_copy_file_name = os.path.splitext(input_audio_audio_file)[0] + "_copy" + os.path.splitext(input_audio_audio_file)[1]
    #callee_copy_file_name = os.path.splitext(callee_audio_file)[0] + "_copy" + os.path.splitext(callee_audio_file)[1]

    #Copy for tests. Should be removed in final version
    input_audio_investig_file_name = os.path.splitext(input_audio_audio_file)[0] + "_investigation" + os.path.splitext(input_audio_audio_file)[1]

    os.system ("cp -f " + input_audio_audio_file + " " + input_audio_copy_file_name)
    os.system ("cp -f " + input_audio_audio_file + " " + input_audio_investig_file_name)
    input_audio_audio_file = input_audio_copy_file_name
    input_audio_music_time_list = []
    #find auto attendant if exists
    input_audio_music_time_list = find_music(input_audio_investig_file_name)
    #
    #detect silence
    input_audio_segments = find_voice_segments(input_audio_investig_file_name, input_audio_music_time_list)
    print(input_audio_segments)

    #make parts by cutting silence pieces
    input_audio_parts = cut_wav_file_on_segments(input_audio_segments, input_audio_audio_file)
    os.remove(input_audio_investig_file_name)
    input_audio_time_text_sychro = [None]*len(input_audio_parts)
    #compand_parts(input_audio_parts)
    ##normalize_parts(input_audio_parts)
    #denoize give small improvement
    denoize_parts(input_audio_parts)

    #Convert to .flac. It is possible to provede .wav to
    #google but I tried to improve quality by .flac.
    #Possibly will revert back in the future.

    i = 0
    for item in input_audio_parts:
        t = threading.Thread(target=translator, args=(i, item, input_audio_time_text_sychro))
        t.daemon = True
        t.start()
        i+=1

    while threading.activeCount() > 1:
        time.sleep(1)
   
    out_data['piece_num'] = len(input_audio_time_text_sychro)
    out_data['pieces'] = []

    for item in input_audio_time_text_sychro:
        for (string, pr) in item[1]:
            out_data['pieces'].append({
            'text': string,
            'start_time': item[0],
            'proba': pr})
            #print(string)

    os.remove(input_audio_copy_file_name);
    for (_,file_segment) in input_audio_parts:
        os.remove(file_segment)
        
    print(json.dumps(out_data, indent=4, ensure_ascii=False))

# #sentiment
# client = language.Client()
# text_content = "Jogging isn't very fun."
# document = client.document_from_text(text_content)
# sentiment_response = document.analyze_sentiment()
# sentiment = sentiment_response.sentiment
# print(sentiment.score)
#
# print(sentiment.magnitude)




