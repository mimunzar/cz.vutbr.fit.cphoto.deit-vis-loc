#!/usr/bin/env python3

import OpenEXR

import src.deit_vis_loc.libs.util as util


class InputAdapter:
    def __init__(self, fpath):
        self.exr_file = OpenEXR.InputFile(fpath)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.exr_file.close()

    def read_header(self):
        return self.exr_file.header()

    def read_channel(self, spec):
        return bytearray(self.exr_file.channel(
            *util.pluck(['channel', 'type'], spec)))


class OutputAdapter:
    def __init__(self, fpath):
        self.fpath    = fpath
        self.exr_file = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.exr_file:
            self.exr_file.close()

    def write_header(self, spec):
        self.header = OpenEXR.Header(*spec['shape'])
        self.header['channels'] = spec['channels']

    def write_channel(self, spec):
        assert self.header, 'Missing header spec'
        self.exr_file = OpenEXR.OutputFile(self.fpath, self.header)
        self.exr_file.writePixels(spec)


def open(fpath, mode):
    if 'r' == mode:
        return InputAdapter(fpath)
    if 'w' == mode:
        return OutputAdapter(fpath)
    raise ValueError(f'Unsupported mode ({mode})')

