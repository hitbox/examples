import argparse
import contextlib
import itertools as it
import math
import os
import time
import unittest

from collections import deque
from operator import attrgetter
from operator import itemgetter
from types import SimpleNamespace

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

# custom events
MOVECURSOR = pygame.event.custom_type()
GRAB = pygame.event.custom_type()
DROP = pygame.event.custom_type()
ROTATE_HOLDING = pygame.event.custom_type()

# pygame key id to two-tuple normalized direction vector
MOVEKEY_DELTA = {
    pygame.K_UP: (0, -1),
    pygame.K_RIGHT: (1, 0),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
}

SIDES = ['top', 'right', 'bottom', 'left']

ADJACENT_NAMES = {
    'top': ('left', 'right'),
    'right': ('top', 'bottom'),
    'bottom': ('right', 'left'),
    'left': ('bottom', 'top'),
}

# "clock-wise" lines
SIDELINES_CW = dict((name, ADJACENT_NAMES[name]) for name in SIDES)

DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

DELTAS_NAMES = dict(zip(DELTAS, SIDES))

get_boundary = attrgetter(*SIDES)

class TestModOffset(unittest.TestCase):

    def test_modoffset(self):
        c = 3
        n = 8
        self.assertEqual(modo(3, n, c), 3)
        self.assertEqual(modo(8, n, c), 3)
        self.assertEqual(modo(9, n, c), 4)
        self.assertEqual(modo(0, n, c), 5)


def modo(a, b, c):
    """
    Returns a modulo b shifted for offset c.
    """
    return c + ((a - c) % (b - c))

def post_movecursor(delta):
    pygame.event.post(pygame.event.Event(MOVECURSOR, delta=delta))

def post_grab():
    pygame.event.post(pygame.event.Event(GRAB))

def post_drop():
    pygame.event.post(pygame.event.Event(DROP))

def post_rotate_holding():
    pygame.event.post(pygame.event.Event(ROTATE_HOLDING))

def nwise(iterable, n=2, fill=None):
    "Take from iterable in `n`-wise tuples."
    iterables = it.tee(iterable, n)
    # advance iterables
    for offset, iterable in enumerate(iterables):
        # advance with for-loop to avoid catching StopIteration manually.
        for _ in zip(range(offset), iterable):
            pass
    return it.zip_longest(*iterables, fillvalue=fill)

def get_rect(*args, **kwargs):
    if not len(args) in (0, 1):
        raise ValueError()
    if len(args) == 1:
        result = args[0]
    else:
        result = pygame.Rect(0,0,0,0)
    for key, val in kwargs.items():
        setattr(result, key, val)
    return result

def wrap(rects):
    boundaries = zip(*map(get_boundary, rects))
    tops, rights, bottoms, lefts = boundaries
    top = min(tops)
    right = max(rights)
    bottom = max(bottoms)
    left = min(lefts)
    width = right - left
    height = bottom - top
    return pygame.Rect(left, top, width, height)

def move_as_one(rects, **kwargs):
    original = wrap(rects)
    destination = original.copy()
    for key, value in kwargs.items():
        setattr(destination, key, value)
    dx = destination.x - original.x
    dy = destination.y - original.y
    for rect in rects:
        rect.x += dx
        rect.y += dy

def align(rects, attrmap):
    """
    :param rects: list of rects.
    :param attrmap: mapping of attribute names.
    """
    # allow for many ways to give mapping but this uses dict interface
    attrmap = dict(attrmap)
    for prevrect, rect in zip(rects, rects[1:]):
        for key, prevkey in attrmap.items():
            setattr(rect, key, getattr(prevrect, prevkey))

def get_color(name, **kwargs):
    # ex.: get_color('red', a=255//2)
    #      for getting 'red' and setting the alpha at the same time
    color = pygame.Color(name)
    for key, val in kwargs.items():
        setattr(color, key, val)
    return color

def cursor_collideitem(cursor, items):
    for item in items:
        for bodyrect in item.body:
            if bodyrect.colliderect(cursor.rect):
                return item

def make_grid_rects(rect_size, rows, columns):
    """
    Return list of rects of given size, arranged in a grid of rows and columns.
    """
    width, height = rect_size
    rects = [
        pygame.Rect(width*i, height*j, width, height)
        for i, j in it.product(range(columns), range(rows))
    ]
    return rects

def draw_grid(surface, cell_size, rows, cols, color, width=1):
    cell_width, cell_height = cell_size
    rect = surface.get_rect()
    for x in range(0, rect.width, cell_width):
        p1 = (x, 0)
        p2 = (x, rect.height)
        pygame.draw.line(surface, color, p1, p2, width)
    for y in range(0, rect.height, cell_height):
        p1 = (0, y)
        p2 = (rect.width, y)
        pygame.draw.line(surface, color, p1, p2, width)

def move_cursor(cursor, delta, items, grid_rect):
    """
    Move inventory cursor. Moves wrap around grid_rect. Moves take whatever
    cursor is holding with it. Cursor warps through items.
    """
    dx, dy = delta

    if cursor.holding:
        # move as if from the topleft of the holding item's body rects
        x, y = cursor.holding.body[0].topleft
    else:
        x, y = cursor.rect.topleft
    x += dx * cursor.rect.width
    y += dy * cursor.rect.height

    # fixup x, y as we go along
    if not cursor.holding:
        colliding_item = cursor_collideitem(cursor, items)
        if colliding_item:
            # warping through items when not holding something
            item_rect = wrap(colliding_item.body)
            if dx < 0:
                x = item_rect.left - cursor.rect.width
            elif dx > 0:
                x = item_rect.right
            if dy < 0:
                y = item_rect.top - cursor.rect.height
            elif dy > 0:
                y = item_rect.bottom

    right, bottom = grid_rect.bottomright
    rects = [cursor.rect]
    if cursor.holding:
        # wrap for dimensions of item, not cursor
        width, height = wrap(cursor.holding.body).size
        right -= width - cursor.rect.width
        bottom -= height - cursor.rect.height
        # add what cursor is holding to list
        rects.extend(cursor.holding.body)

    # move everything as one to keep cursor relative position
    x = modo(x, right, grid_rect.left)
    y = modo(y, bottom, grid_rect.top)
    move_as_one(rects, x=x, y=y)

def rotate_rects(rects):
    """
    Rotate rects 90 degrees.

    Rects are grouped into rows by their y attribute, in a nested list.
    Transpose the list and use its order to reflow the rects' positions from
    topleft.
    """
    x_pos = attrgetter('x')
    y_pos = attrgetter('y')
    grouped = it.groupby(sorted(rects, key=y_pos), key=y_pos)
    table = [ list(items) for key, items in grouped ]
    rotated_table = list(zip(*table))
    wrapped = wrap(rects)
    left, top = wrapped.topleft
    for row in rotated_table:
        row = sorted(row, key=x_pos)
        row[0].left = left
        for r1, r2 in nwise(row):
            r1.top = top
            if r2:
                r2.topleft = (r1.right, top)
        top = r1.bottom

def rotate_holding(cursor, grid_rect):
    rotate_rects(cursor.holding.body + [cursor.rect])

def jump_cursor(cursor, delta, items, grid_rect):
    dx, dy = delta

    if not (dx == 0 or dy == 0):
        raise ValueError()

    if cursor.holding:
        rect = wrap(cursor.holding.body)
    else:
        rect = cursor.rect

    other_bodies = [
        wrap(item.body) for item in items
        if cursor.holding and cursor.holding is not item
    ]

    BOUNDARY_NAME = DELTAS_NAMES[delta]
    line_attrs = ADJACENT_NAMES[BOUNDARY_NAME]
    # TODO
    # - left off here
    # - need the line projection test thingy like shrinkwrap

def place_item(cell_size, item, items):
    width, height = cell_size
    while True:
        if not any(
            rect.colliderect(other_rect)
            for other in items
            for other_rect in other.body
            for rect in item.body
        ):
            break

def run(
    output_string = None,
):
    clock = pygame.time.Clock()
    fps = 60

    item_font = pygame.font.SysFont('monospace', 40)

    window = pygame.display.get_surface()
    frame = window.get_rect()

    background = window.copy()

    help_ = SimpleNamespace(
        color = 'ghostwhite',
        font = pygame.font.SysFont('monospace', 40),
        frame = frame.inflate((-min(frame.size)//32, ) * 2)
    )
    help_.images = [
        help_.font.render('Arrow keys to move', True, help_.color),
        help_.font.render('Return or Space to grab', True, help_.color),
        help_.font.render('Tab to rotate', True, help_.color),
    ]
    help_.rects = [image.get_rect() for image in help_.images]
    help_.rects[0].topright = help_.frame.topright
    align(help_.rects, {'topright':'bottomright'})

    for image, rect in zip(help_.images, help_.rects):
        background.blit(image, rect)

    grid = SimpleNamespace(
        rows = 7,
        cols = 11,
        cell_size = (50,)*2,
    )
    real_size = (grid.cols * grid.cell_size[0], grid.rows * grid.cell_size[1])
    image_size = tuple(map(lambda x: x+1, real_size))
    grid.image = pygame.Surface(image_size, flags = pygame.SRCALPHA)
    draw_grid(grid.image, grid.cell_size, grid.rows, grid.cols, 'ghostwhite')
    grid.rect = get_rect(size=real_size, center=frame.center)

    cursor = SimpleNamespace(
        rect = pygame.Rect(grid.rect.topleft, grid.cell_size),
        holding = None,
        inflation = it.cycle(
            (framesize,)*2 for size in it.chain(range(10), range(10,-1,-1))
            for framesize in it.repeat(size, 2)
        ),
    )

    # make items
    pistol = SimpleNamespace(
        body = make_grid_rects(grid.cell_size, 2, 3),
        color = 'red',
    )
    move_as_one(pistol.body, topleft=grid.rect.topleft)

    rifle = SimpleNamespace(
        body = make_grid_rects(grid.cell_size, 1, 9),
        color = 'burlywood4',
    )
    move_as_one(rifle.body, top=pistol.body[-1].bottom, left=grid.rect.left)

    grenade = SimpleNamespace(
        body = make_grid_rects(grid.cell_size, 2, 1),
        color = 'darkgreen',
    )
    move_as_one(grenade.body, top=rifle.body[-1].bottom, left=grid.rect.left)

    chicken_egg = SimpleNamespace(
        body = make_grid_rects(grid.cell_size, 1, 1),
        color = 'oldlace',
    )
    move_as_one(chicken_egg.body, top=grenade.body[-1].bottom, left=grid.rect.left)

    items = [
        pistol,
        rifle,
        grenade,
        chicken_egg,
    ]

    frame_queue = deque()
    frame_num = 0
    running = True
    while running:
        if frame_queue:
            while frame_queue and clock.get_fps() > fps:
                frame_image = frame_queue.popleft()
                path = output_string.format(frame_num)
                pygame.image.save(frame_image, path)
                frame_num += 1
        elapsed = clock.tick(fps)

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, ):
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                elif event.key in MOVEKEY_DELTA:
                    # key to move event
                    post_movecursor(MOVEKEY_DELTA[event.key])
                elif event.key == pygame.K_TAB:
                    post_rotate_holding()
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if cursor.holding:
                        post_drop()
                    else:
                        post_grab()
            elif event.type == MOVECURSOR:
                # move cursor with wrapping
                move_cursor(cursor, event.delta, items, grid.rect)
            elif event.type == GRAB:
                # grab item
                item = cursor_collideitem(cursor, items)
                if item:
                    cursor.holding = item
            elif event.type == DROP:
                # drop item
                cursor.holding = None
            elif event.type == ROTATE_HOLDING:
                if cursor.holding:
                    rotate_holding(cursor, grid.rect)
        # draw
        window.blit(background, (0,)*2)
        window.blit(grid.image, grid.rect)
        # draw - items
        for item in items:
            for rect in item.body:
                pygame.draw.rect(window, item.color, rect)
        # draw - cursor
        if not cursor.holding:
            item = cursor_collideitem(cursor, items)
            if item:
                cursor_rect = wrap(item.body)
            else:
                cursor_rect = cursor.rect
            inflatesize = next(cursor.inflation)
            cursor_rect = cursor_rect.inflate(*inflatesize)
            pygame.draw.rect(window, 'yellow', cursor_rect, 1)
        if False:
            # draw - fps
            image = help_.font.render(f'{clock.get_fps():.2f}', True, 'ghostwhite')
            last = window.blit(image, image.get_rect(bottomright=help_.frame.bottomright))
            # draw - len(frame_queue)
            image = help_.font.render(f'{len(frame_queue)=:03d}', True, 'ghostwhite')
            last = window.blit(image, image.get_rect(bottomright=last.topright))
            # draw - frame_num
            image = help_.font.render(f'{frame_num}', True, 'ghostwhite')
            last = window.blit(image, image.get_rect(bottomright=last.topright))
        #
        pygame.display.flip()
        if output_string:
            frame_queue.append(window.copy())

def start(options):
    pygame.font.init()
    window = pygame.display.set_mode(options.size)
    frame = window.get_rect()
    run(output_string=options.output)

def sizetype(string):
    """
    Parse string into a tuple of integers.
    """
    size = tuple(map(int, string.replace(',', ' ').split()))
    if len(size) == 1:
        size += size
    return size

def cli():
    """
    Inventory
    """
    # saw this:
    # https://www.reddit.com/r/pygame/comments/xasi84/inventorycrafting_system/
    # TODO:
    # [ ] Like Resident Evil 4 in 2d
    # [X] grab / drop
    # [ ] Auto arrange with drag+drop animations
    # [X] Rotate items
    # [ ] Stacking?
    # [ ] Combine to new item?
    # [ ] Stealing minigame. Something chases or attacks your cursor.
    # [ ] Moving through items are a way of jumping, could be part of gameplay.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size',
        default = '800,800',
        type = sizetype,
        help = 'Screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--output',
        help = 'Format string for frame output.',
    )
    args = parser.parse_args()
    start(args)

if __name__ == '__main__':
    cli()
