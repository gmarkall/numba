from array import array
import numpy as np
from math import sqrt
from numba import jit

import os, re, array
from subprocess import Popen, PIPE, STDOUT


def mplayer(Image, fn='tv://', options=''):
    f = os.popen('mplayer -really-quiet -noframedrop ' + options + ' ' 
                 '-vo yuv4mpeg:file=/dev/stdout 2>/dev/null </dev/null ' + fn)
    hdr = f.readline()
    m = re.search('W(\d+) H(\d+)', hdr)
    w, h = int(m.group(1)), int(m.group(2))
    while True:
        hdr = f.readline()
        if hdr != 'FRAME\n':
            break
        yield Image(w, h, fromfile=f)
        f.read(w*h/2) # Color data

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
        img.T.tofile(self.mplayer.stdin)

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
        fn = sys.argv[1]
    else:
        fn = 'sobel_test.avi -vf scale=640:480 -benchmark'

    start = start0 = time()
    for fcnt, img in enumerate(mplayer(get_image, fn)):
        view(sobel_magnitude(img))
        print 1.0 / (time() - start), 'fps, ', (fcnt-2) / (time() - start0), 'average fps'
        start = time()
        if fcnt==2:
            start0 = time()
