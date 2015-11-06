# This demo is based on the PyPy Sobel demo, originally obtained from:
#
# https://bitbucket.org/pypy/extradoc/src/df759011beb46610c02d065ae159263817a2aa0b/talk/dls2012/benchmarks/image/?at=extradoc
#
# To run the demo with the provided video file, use:
#
# $ python sobel.py
#
# To run with an alternative file, arguments to mplayer can be specified on the
# command line, e.g.:
#
# $ python sobel.py tv://
#
# to run in realtime from a webcam.

from __future__ import print_function, division
import numpy as np
import os
import re

from math import sqrt
from numba import jit
from subprocess import Popen, PIPE, STDOUT


def mplayer(args):
    f = os.popen('mplayer -really-quiet -noframedrop '
                 '-vo yuv4mpeg:file=/dev/stdout 2>/dev/null </dev/null ' + args)
    hdr = f.readline()
    m = re.search('W(\d+) H(\d+)', hdr)
    w, h = int(m.group(1)), int(m.group(2))
    while True:
        hdr = f.readline()
        if hdr != 'FRAME\n':
            break
        yield get_image(w, h, fromfile=f)
        f.read(w*h//2) # Color data


class MplayerViewer(object):
    def __init__(self):
        self.width = self.height = None

    def view(self, img):
        if not self.width:
            w, h = img.shape[0], img.shape[1]
            self.mplayer = Popen(['mplayer', '-', '-benchmark',
                                  '-demuxer', 'rawvideo',
                                 '-rawvideo', 'w=%d:h=%d:format=y8' % (w, h),
                                 '-really-quiet'],
                                 stdin=PIPE, stdout=PIPE, stderr=PIPE)
            self.width = w
            self.height = h
        assert self.width == img.shape[0]
        assert self.height == img.shape[1]
        write_image(img, self.mplayer.stdin)

default_viewer = MplayerViewer()

def view(img):
    default_viewer.view(img)


def get_image(w, h, fromfile):
    return np.fromfile(fromfile, dtype=np.uint8, count=w*h).reshape((h, w)).transpose()

def write_image(img, f):
    img.T.tofile(f)


@jit
def sobel_magnitude(a):
    b = np.zeros_like(a)
    for y in xrange(1, a.shape[1]-1):
        for x in xrange(1, a.shape[0]-1):
            dx = -1.0 * a[x-1, y-1] + 1.0 * a[x+1, y-1] + \
                 -2.0 * a[x-1, y]   + 2.0 * a[x+1, y] + \
                 -1.0 * a[x-1, y+1] + 1.0 * a[x+1, y+1]
            dy = -1.0 * a[x-1, y-1] -2.0 * a[x, y-1] -1.0 * a[x+1, y-1] + \
                  1.0 * a[x-1, y+1] +2.0 * a[x, y+1] +1.0 * a[x+1, y+1]
            b[x, y] = min(int(sqrt(dx*dx + dy*dy) / 4.0), 255)

    return b


if __name__ == '__main__':
    import sys
    from time import time

    if len(sys.argv) > 1:
        mp_args = ' '.join(sys.argv[1:])
    else:
        mp_args = 'sobel_test.avi -vf scale=640:480 -benchmark'

    start = start0 = time()
    for fcnt, img in enumerate(mplayer(mp_args)):
        view(sobel_magnitude(img))
        fps = 1.0 / (time() - start)
        avg = (fcnt-2) / (time() - start0)
        print('%s fps, %s average fps' % (fps, avg))
        start = time()
        if fcnt==2:
            start0 = time()
