import argparse
import contextlib
import itertools as it
import operator
import os
import unittest

from collections import deque
from functools import partial
from types import SimpleNamespace

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

CUTRECT = pygame.event.custom_type()

SIDES = ['top', 'right', 'bottom', 'left']
OPPOSITE = {s: SIDES[i % len(SIDES)] for i, s in enumerate(SIDES, start=2)}

# index-aligned side name to its appropriate dimension
SIDEDIMENSION = ['height', 'width'] * 2

SIDEDIMENSION = dict(zip(SIDES, ['height', 'width']*2))

class TestResizeSide(unittest.TestCase):

    def setUp(self):
        self.r = pygame.Rect(0, 0, 10, 10)

    def test_resize_side_top(self):
        resize_side_ip(self.r, 'top', 2)
        self.assertEqual(self.r, pygame.Rect(0, 2, 10, 8))

    def test_resize_side_left(self):
        resize_side_ip(self.r, 'left', 2)
        self.assertEqual(self.r, pygame.Rect(2, 0, 8, 10))

    def test_resize_side_right(self):
        resize_side_ip(self.r, 'right', 2)
        self.assertEqual(self.r, pygame.Rect(0, 0, 12, 10))

    def test_resize_side_bottom(self):
        resize_side_ip(self.r, 'bottom', 2)
        self.assertEqual(self.r, pygame.Rect(0, 0, 10, 12))


class TestCutRect(unittest.TestCase):

    w = h = 10

    def setUp(self):
        self.r = pygame.Rect(0, 0, self.w, self.h)

    def test_cut_rect_ip_top(self):
        remaining = cut_rect_ip(self.r, 'top', 2)
        self.assertEqual(self.r, pygame.Rect(0, 2, self.w, self.h - 2))
        self.assertEqual(remaining, pygame.Rect(0, 0, self.w, 2))

    def test_cut_rect_ip_right(self):
        remaining = cut_rect_ip(self.r, 'right', 2)
        self.assertEqual(self.r, pygame.Rect(0, 0, self.w - 2, self.h))
        self.assertEqual(remaining, pygame.Rect(self.w - 2, 0, 2, self.w))

    def test_cut_rect_ip_bottom(self):
        remaining = cut_rect_ip(self.r, 'bottom', 2)
        self.assertEqual(self.r, pygame.Rect(0, 0, self.w, self.h - 2))
        self.assertEqual(remaining, pygame.Rect(0, self.h - 2, self.w, 2))

    def test_cut_rect_ip_left(self):
        remaining = cut_rect_ip(self.r, 'left', 2)
        self.assertEqual(self.r, pygame.Rect(2, 0, self.w - 2, self.h))
        self.assertEqual(remaining, pygame.Rect(0, 0, 2, self.h))


def get_rect(*args, **kwargs):
    """
    :param *args:
        Optional rect used as base. Otherwise new (0,)*4 rect is created.
    :param kwargs:
        Keyword arguments to set on new rect.
    """
    if len(args) not in (0, 1):
        raise ValueError()
    if len(args) == 1:
        result = args[0].copy()
    else:
        result = pygame.Rect(0,0,0,0)
    for key, val in kwargs.items():
        setattr(result, key, val)
    return result

def resize_side_ip(rect, side, amount):
    """
    Resize a rect's side, in place, accounting for the sides top and left
    moving the rect.
    """
    dimension = SIDEDIMENSION[side]
    dimension_amount = getattr(rect, dimension)
    if side in ('top', 'left'):
        setattr(rect, side, (getattr(rect, side) + amount))
        dimension_amount -= amount
    else:
        dimension_amount += amount
    setattr(rect, dimension, dimension_amount)

def cut_rect_ip(rect, side, amount):
    """
    Cut a rect side by amount, in place; and return a new rect that fits inside
    the cut out part.
    """
    result = rect.copy()
    # size new rect
    dimension = SIDEDIMENSION[side]
    setattr(result, dimension, amount)
    # align new rect to same side being cut
    setattr(result, side, getattr(rect, side))
    # in place resize rect for cut
    resize_amount = amount
    if side in ('right', 'bottom'):
        resize_amount = -resize_amount
    resize_side_ip(rect, side, resize_amount)
    return result

def post_cutrect():
    pygame.event.post(pygame.event.Event(CUTRECT))

def run(
    cutamount = None,
    scale = None,
    output_string = None,
):
    clock = pygame.time.Clock()
    fps = 60
    screen = pygame.display.get_surface()
    frame = screen.get_rect()
    gui_frame = frame.inflate((-10,)*2)
    gui_font = pygame.font.SysFont('arial', 30)
    rect_font = pygame.font.SysFont('monospace', 10)

    if scale is None or scale == 1:
        buffer = screen.copy()
    else:
        buffer = pygame.Surface(tuple(map(lambda d: d / scale, frame.size)))
    sspace = buffer.get_rect()

    colors = [
        'red',
        'yellow',
        'green',
        'blue',
        'aquamarine4',
        'chocolate',
    ]
    drawrects = []
    cutsides = it.cycle(SIDES)
    layout = None
    cut_side = None

    def on_cutrect(event):
        nonlocal cut_side
        nonlocal cutamount
        nonlocal layout
        pygame.time.set_timer(CUTRECT, 500)
        if not drawrects:
            cut_side = next(cutsides)
            layout = sspace.inflate((-20,)*2)
            layout.center = sspace.center
            drawrects.append(layout)
            if cutamount is None:
                cutamount = min(layout.size) // 12
        elif drawrects:
            if len(drawrects) <= 5:
                drawrects.append(cut_rect_ip(layout, cut_side, cutamount))
            else:
                pygame.time.set_timer(CUTRECT, 0)
                post_cutrect()
                drawrects.clear()

    handlers = {
        CUTRECT: on_cutrect,
    }
    post_cutrect()

    frame_num = 0
    frame_queue = deque()
    running = True
    while running:
        # tick and frame saving
        if output_string and frame_queue:
            # save frame from queue until fps is met or run out of frames
            while frame_queue and clock.get_fps() > fps:
                frame_image = frame_queue.popleft()
                path = output_string.format(frame_num)
                pygame.image.save(frame_image, path)
                frame_num += 1
        elapsed = clock.tick(fps)
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # stop main loop after this frame
                running = False
            elif event.type in handlers:
                handlers[event.type](event)
        # draw
        buffer.fill('black')
        for number, (rect, color) in enumerate(zip(drawrects, colors)):
            pygame.draw.rect(buffer, color, rect, 1)
            if number == 0:
                string = 'Layout'
            else:
                string = f'R{number}'
            text = rect_font.render(string, False, color)
            buffer.blit(text, text.get_rect(center=rect.center))
        # draw - scale for final
        if scale is None or scale == 1:
            screen.blit(buffer, (0,)*2)
        else:
            pygame.transform.scale(buffer, frame.size, screen)
        # draw - gui text
        text = gui_font.render(f'Cutting {cut_side}', True, 'ghostwhite')
        screen.blit(text, text.get_rect(bottomright=gui_frame.bottomright))
        pygame.display.flip()
        if output_string:
            frame_queue.append(screen.copy())

def start(options):
    """
    Initialize and start run loop. Bridge between options and main loop.
    """
    pygame.font.init()
    screen_size = tuple(map(lambda d: options.scale * d, options.size))
    pygame.display.set_mode(screen_size)
    run(
        scale = options.scale,
        cutamount = options.cut,
        output_string = options.output,
    )

def sizetype(string):
    """
    Parse string into a tuple of integers.
    """
    size = tuple(map(int, string.replace(',', ' ').split()))
    if len(size) == 1:
        size += size
    return size

def cli():
    # https://halt.software/dead-simple-layouts/
    # TODO
    # [X] generic function taking sidename to cut
    # [ ] non-inplace versions? does that make sense? what's the use?
    # [ ] resize bordering rects like tiling window manager
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size',
        default = '200,200',
        type = sizetype,
        help = 'Screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--scale',
        default = '4',
        type = int,
        help = 'Scale of size, final screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--cut',
        type = int,
        help = 'Cut width. Default: %(default)s',
    )
    parser.add_argument(
        '--output',
        help = 'Format string for frame output.',
    )
    args = parser.parse_args()
    start(args)

if __name__ == '__main__':
    cli()
