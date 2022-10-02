import argparse
import contextlib
import itertools as it
import json
import math
import operator
import os
import random
import string
import unittest

from abc import ABC
from abc import abstractmethod
from collections import deque
from functools import partial
from operator import attrgetter
from types import SimpleNamespace

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

SWITCH = pygame.event.custom_type()

getx = attrgetter('x')

class TestBounce(unittest.TestCase):

    def test_bounce(self):
        cat = ''.join
        self.assertEqual(cat(bounce('a')), 'a')
        self.assertEqual(cat(bounce('ab')), 'ab')
        self.assertEqual(cat(bounce('abc')), 'abcb')
        self.assertEqual(cat(bounce('abcd')), 'abcdcb')


class Tab:

    def __init__(self, rect, image):
        self.rect = rect
        self.image = image


class TabStyle:

    def __init__(self, rect, image, color, fill_slot):
        self.rect = rect
        self.image = image
        self.color = color
        self.fill_slot = fill_slot

    @classmethod
    def drawargs(cls, tab, slot, hovering, dragging):
        fill_slot = None
        if tab is hovering or tab is dragging:
            rect = tab.rect
            if tab is hovering:
                color = 'blue'
            elif tab is dragging:
                color = 'green'
                fill_slot = None
        else:
            color = 'red'
            rect = get_rect(tab.rect, center=slot.center)
        return cls(rect, tab.image, color, )


class Container:
    """
    Tab Container
    """

    def __init__(self, rect, tabs, slots, sortattr=None):
        self.rect = rect
        self.tabs = tabs
        self.slots = slots
        if sortattr is None:
            sortattr = 'rect.x'
        self.sortby = attrgetter(sortattr)

    def as_sorted_pairs(self):
        """
        Sort tabs and pair against (unsorted) slots.
        """
        return zip(sorted(self.tabs, key=self.sortby), self.slots)

    def commit_sorted(self):
        """
        Commit the tabs to the slot they're paired against after sorting.
        """
        for tab, slot in self.as_sorted_pairs():
            tab.rect.center = slot.center


class BaseState(ABC):

    @abstractmethod
    def handle(self, event):
        pass


class DispatchEventMixin:

    def handle(self, event):
        """
        Dispatch to event handler by naming convention.
        """
        name = pygame.event.event_name(event.type)
        handler = getattr(self, f'on_{name.lower()}', None)
        if handler:
            handler(event)


class TabBarState(DispatchEventMixin, BaseState):
    "Tab Bar"
    # [ ] rects inside a container rect
    # [ ] smaller rects are draggable
    # [ ] rects reorder themselves against dragging rect
    # [ ] rects move in an animated fashion

    def __init__(self, output=None):
        self.output = output
        self.frame_number = 0
        self.screen = pygame.display.get_surface()
        self.frame = self.screen.get_rect()
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.tick_accumulator = 0
        self.gui_font = pygame.font.SysFont('Arial', 32)
        self.tab_font = pygame.font.SysFont('Arial', 20)

        ntabs = 5
        container_rect = get_rect(
            self.frame,
            width = self.frame.width * .9,
            height = self.frame.height * .10,
            center = self.frame.center,
        )
        self.tabbar = make_tabbar(
            container_rect = container_rect,
            tab_width = container_rect.width // ntabs,
            tab_margin = min(container_rect.size) * .20,
            sortattr = 'rect.centerx'
        )
        for n, tab in enumerate(self.tabbar.tabs, start=1):
            tab.image = self.tab_font.render(f'Tab {n}', True, 'ghostwhite')

        self.hovering = None
        self.dragging = None

    def on_quit(self, event):
        switch(None)

    def on_keydown(self, event):
        if event.key in (pygame.K_ESCAPE, pygame.K_q):
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def on_mousemotion(self, event):
        if self.dragging:
            self.dragging.rect.move_ip((event.rel[0], 0))
        else:
            for tab in self.tabbar.tabs:
                if tab.rect.collidepoint(event.pos):
                    self.hovering = tab
                    break
            else:
                self.hovering = None

    def on_mousebuttondown(self, event):
        if self.hovering:
            self.dragging = self.hovering
            self.hovering = None

    def on_mousebuttonup(self, event):
        if self.dragging:
            self.tabbar.commit_sorted()
        self.dragging = None

    def draw(self):
        self.screen.fill('black')
        # draw container
        pygame.draw.rect(self.screen, 'ghostwhite', self.tabbar.rect, 1)
        # draw tabs
        for tab, slot in self.tabbar.as_sorted_pairs():
            if tab is self.dragging:
                rect = tab.rect
                pygame.draw.rect(self.screen, 'green', slot, 1)
                pygame.draw.rect(self.screen, 'chartreuse4', slot, 0)
            else:
                rect = get_rect(tab.rect, center=slot.center)
            text_rect = tab.image.get_rect(center=rect.center)
            self.screen.blit(tab.image, text_rect)
            pygame.draw.rect(self.screen, 'ghostwhite', rect, 1)

        pygame.display.flip()
        if self.output:
            self.clock.tick(self.fps)
            filename = self.output.format(self.frame_number)
            pygame.image.save(self.screen, filename)
            self.frame_number += 1


def get_rect(*rect_arg_or_nothing, **rectattrs):
    nargs = len(rect_arg_or_nothing)
    if nargs not in (0, 1):
        raise ValueError()
    elif nargs == 0:
        result = pygame.Rect((0,)*4)
    else:
        result = pygame.Rect(rect_arg_or_nothing)
    for key, value in rectattrs.items():
        setattr(result, key, value)
    return result

def bounce(iterable):
    """
    Yield the items forwardly and backwards without repeating ends.
    a, b, c -> a b c b a
    """
    # the motivation behind this is the way CSS unpacks shorthand values, often
    # around four properties, like top, right, bottom, left:
    # 1-value: 1 1 1 1
    # 2-value: 1 2 1 2
    # 3-value: 1 2 3 2
    # 4-value: 1 2 3 4
    # [ ] itertools.cycle works for n-values 1 or 2
    # [ ] three value is the weird one
    # [ ] four value is straight unpacking in the max four value case
    # [ ] what about higher values? maybe all the points around a rect?
    #     midtop, topright, midright, bottomright, ..., topleft
    stack = []
    for index, item in enumerate(iterable):
        yield item
        if index > 0:
            stack.append(item)
    if stack:
        stack.pop()
        while stack:
            yield stack.pop()

def assign_slots_ip(rects, slots, attr='topleft'):
    """
    """
    getrectattr = attrgetter(attr)
    for rect, slot in zip(rects, slots):
        setattr(rect, attr, getrectattr(slot))

def subdivide_rect(rect, width, height):
    ys = range(rect.top, rect.bottom, height)
    xs = range(rect.left, rect.right, width)
    for y, x in it.product(ys, xs):
        yield pygame.Rect(x, y, width, height)

def make_tabbar(container_rect, tab_width, tab_margin, sortattr=None):
    slots = list(subdivide_rect(container_rect, tab_width, container_rect.height))
    tabs = [
        Tab(
            rect = get_rect(
                size = (
                    tab_width - tab_margin * 2,
                    container_rect.height - tab_margin * 2
                ),
                center = slot.center
            ),
            image = None,
        )
        for slot in slots
    ]
    container = Container(container_rect, tabs, slots, sortattr=sortattr)
    return container

def switch(state, **data):
    event = pygame.event.Event(SWITCH, state=state, **data)
    pygame.event.post(event)

def run(state):
    next_ = state
    running = True
    while running:
        if next_ is not None:
            # state.leave(...)?
            # next_.enter(...)?
            state = next_
            next_ = None
            calls = [
                getattr(state, attr)
                for attr in ['update', 'draw']
                if hasattr(state, attr)
            ]
        for event in pygame.event.get():
            if event.type != SWITCH:
                state.handle(event)
            else:
                # states indicate that they want the engine to stop by
                # switching to None state.
                if event.state is None:
                    running = False
                else:
                    next_ = event.state
        for call in calls:
            call()

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
        default = '800,800',
        type = sizetype,
        help = 'Screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--output',
        help = 'Format string for frame output. ex: path/to/frames{:05d}.bmp',
    )
    args = parser.parse_args()

    pygame.font.init()
    pygame.display.set_mode(args.size)

    state = TabBarState(output=args.output)
    run(state)

if __name__ == '__main__':
    cli()
