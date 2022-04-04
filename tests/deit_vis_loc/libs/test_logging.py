#!/usr/bin/env python3

import src.deit_vis_loc.libs.logging as logging


def test_progress_bar():
    assert logging.progress_bar(1, 1, 0) == '[ ] 0/1'
    assert logging.progress_bar(1, 1, 1) == '[#] 1/1'
    assert logging.progress_bar(1, 1, 2) == '[#] 1/1'

    assert logging.progress_bar(5, 1, 0)    == '[     ] 0/1'
    assert logging.progress_bar(5, 1, 0.33) == '[##   ] 0.33/1'
    assert logging.progress_bar(5, 1, 1)    == '[#####] 1/1'

    assert logging.progress_bar(10, 5, 0) == '[          ] 0/5'
    assert logging.progress_bar(10, 5, 1) == '[##        ] 1/5'
    assert logging.progress_bar(10, 5, 5) == '[##########] 5/5'


def test_format_fraction():
    assert logging.format_fraction(1, 1)   == '1/1'
    assert logging.format_fraction(1, 10)  == ' 1/10'
    assert logging.format_fraction(1, 100) == '  1/100'


def test_make_progress_formatter():
    f = logging.make_progress_formatter(bar_width=1, total=1)
    assert f(stage='Foo', curr=0, speed=0, loss=0) == \
            '            Foo: [ ] 0/1  (0.00 loss, 0.00 im/s)'
    assert f(stage='Foo', curr=1, speed=0.5, loss=0.5) == \
            '            Foo: [#] 1/1  (0.50 loss, 0.50 im/s)'
    assert f(stage='FooFoo', curr=1, speed=0.5, loss=0.5) == \
            '         FooFoo: [#] 1/1  (0.50 loss, 0.50 im/s)'
    assert f(stage='Foo',    curr=1, speed=1000.5, loss=1000.5) == \
            '            Foo: [#] 1/1  (1000.50 loss, 1000.50 im/s)'


def test_make_ims_sec():
    ims_sec = logging.make_ims_sec(lambda: 0)
    assert ims_sec(1, lambda: 1) == 1   # 1 seconds diff
    assert ims_sec(5, lambda: 6) == 1   # 5 seconds diff
    assert ims_sec(5, lambda: 6) == 5e6 # 0 seconds diff

