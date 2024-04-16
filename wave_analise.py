import wave
import numpy

waveFile = wave.open('./archive/data/0_a.wav', 'rb')
nframes = waveFile.getnframes()
wavFrames = waveFile.readframes(nframes)
ys = numpy.fromstring(wavFrames, dtype=numpy.int16)

print('finish')