import argparse
import contextlib
import itertools as it
import math
import operator
import os
import random
import string
import unittest

from collections import deque
from functools import partial
from operator import attrgetter
from operator import itemgetter
from operator import mul
from pprint import pprint

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

STEP = pygame.event.custom_type()

def cubic_bezier(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + t*p1*(3*(1-t)**2) + p2*(3*(1-t)*t**2) + p3*t**3

def nwise(iterable, n=2):
    "Take from iterable in `n`-wise tuples."
    iterables = it.tee(iterable, n)
    # advance iterables for offsets
    for offset, iterable in enumerate(iterables):
        # advance with for-loop to avoid catching StopIteration manually.
        for _ in zip(range(offset), iterable):
            pass
    return zip(*iterables)

def bendangle(p1, middle, p2):
    return abs(
        math.atan2(p1.y - middle.y, p1.x - middle.x)
        -
        math.atan2(p2.y - middle.y, p2.x - middle.x)
    )

def sizetype(string):
    "Parse string into a tuple of integers."
    size = tuple(map(int, string.replace(',', ' ').split()))
    if len(size) == 1:
        size += size
    return size

def cli():
    "Tab bar"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size',
        default = '1024,',
        type = sizetype,
        help = 'Screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--output',
        help = 'Format string for frame output. ex: path/to/frames{:05d}.bmp',
    )
    args = parser.parse_args()

    pygame.font.init()
    clock = pygame.time.Clock()
    fps = 60
    screen = pygame.display.set_mode(args.size)
    gui_font = pygame.font.SysFont('monospace', 24)
    small_font = pygame.font.SysFont('monospace', 18)
    frame = screen.get_rect()
    gui_frame = frame.inflate((-min(frame.size)*.5,)*2)
    viewbox = frame.inflate((-min(frame.size)*.8,)*2)

    control_points = [
        pygame.Vector2(viewbox.topleft),
        pygame.Vector2(frame.topright),
        pygame.Vector2(frame.bottomleft),
        pygame.Vector2(viewbox.bottomright),
    ]

    animate = False
    maxsubdivisions = 2 + 3
    n = 0
    tpoints = []
    calctimes = []
    calcpoints = []

    def eval_animate():
        nonlocal n
        nonlocal animate
        n = 0
        if animate:
            pygame.time.set_timer(STEP, 200)
        else:
            pygame.time.set_timer(STEP, 0)
            calculate_tpoints()

    def calculate_tpoints():
        nonlocal n
        nonlocal tpoints
        nonlocal calctimes
        nonlocal calcpoints
        while n <= maxsubdivisions:
            if n == 0:
                tpoints = [(t, cubic_bezier(t, *control_points)) for t in [0,1]]
                calctimes = [.5]
                n += 1
                if animate:
                    break
            elif calctimes:
                calc_t = calctimes.pop()
                tpoints = sorted(tpoints + [(calc_t, cubic_bezier(calc_t, *control_points))])
                if animate:
                    break
            elif n < maxsubdivisions:
                # NOTE:
                # - math.dist was tried between pairs, here. it does ok.
                angles = [
                    (t1, p1, t2, p2, math.pi - bendangle(p1, mp, p2))
                    for (t1, p1), (mt, mp), (t2, p2) in nwise(tpoints, 3)
                ]
                sortedpairs = sorted(angles, key=itemgetter(4), reverse=True)
                calctimes = [(t1 + (t2 - t1) / 2) for t1, p1, t2, p2, dist in sortedpairs]
                calcpoints = [p for t1, p1, t2, p2, dist in sortedpairs for p in [p1, p2]]
                n += 1
                if animate:
                    break
            else:
                calcpoints.clear()
                break

    eval_animate()
    hovering = None
    dragging = None
    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                elif event.key == pygame.K_SPACE:
                    # toggle animation
                    animate = not animate
                    eval_animate()
                elif event.key == pygame.K_r:
                    # replay
                    eval_animate()
                elif event.key == pygame.K_a:
                    # more subdivision
                    maxsubdivisions += 1
                    eval_animate()
                elif event.key == pygame.K_z:
                    # less subdivision
                    maxsubdivisions -= 1
                    eval_animate()
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dragging.update(event.pos)
                    eval_animate()
                else:
                    for i, p in enumerate(control_points):
                        if math.dist(p, event.pos) < 50:
                            hovering = p
                            break
                    else:
                        hovering = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    if hovering:
                        dragging = hovering
                        hovering = None
                elif event.button == pygame.BUTTON_WHEELUP:
                    maxsubdivisions += 1
                    eval_animate()
                elif event.button == pygame.BUTTON_WHEELDOWN:
                    maxsubdivisions -= 1
                    eval_animate()
            elif event.type == pygame.MOUSEBUTTONUP:
                # drop point
                if event.button == pygame.BUTTON_LEFT:
                    dragging = None
            elif event.type == STEP:
                calculate_tpoints()
        #
        screen.fill('black')
        # text info
        image = gui_font.render(f'{maxsubdivisions=}', True, 'ghostwhite')
        rect = image.get_rect(bottomright=gui_frame.bottomright)
        screen.blit(image, rect)
        image = gui_font.render(f'{animate=}', True, 'ghostwhite')
        rect = image.get_rect(bottomright=rect.topright)
        screen.blit(image, rect)
        image = gui_font.render(f'{n=}', True, 'ghostwhite')
        rect = image.get_rect(bottomright=rect.topright)
        screen.blit(image, rect)
        for p in control_points:
            if p is hovering:
                radius = 20
                color = 'green'
            else:
                radius = 4
                color = 'blue'
            pygame.draw.circle(screen, color, p, radius, 1)
        # handle lines
        pygame.draw.line(screen, 'green', control_points[0], control_points[1], 1)
        pygame.draw.line(screen, 'green', control_points[2], control_points[3], 1)
        # lines between time points and points needing subdivision
        for (t1, p1), (t2, p2) in it.pairwise(tpoints):
            pygame.draw.line(screen, 'red', p1, p2, 1)
            if p1 in calcpoints:
                pygame.draw.circle(screen, 'peachpuff3', p1, 8, 1)
            if p2 in calcpoints:
                pygame.draw.circle(screen, 'peachpuff3', p2, 8, 1)
        pygame.display.flip()
        if args.output:
            filename = args.output.format(frame)
            pygame.image.save(screen, filename)
            frame += 1
            clock.tick(fps)

if __name__ == '__main__':
    cli()
