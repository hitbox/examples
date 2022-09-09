import argparse
import contextlib
import itertools as it
import operator
import os
import random
import unittest

from collections import deque
from functools import partial

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

REPAINT = pygame.event.custom_type()
MOVESNAKE = pygame.event.custom_type()
GENERATEFOOD = pygame.event.custom_type()
EATFOOD = pygame.event.custom_type()

SIDES = ['top', 'right', 'bottom', 'left']
ADJACENT_SIDE_CCW = {side: SIDES[i % len(SIDES)] for i, side in enumerate(SIDES, start=-1)}
ADJACENT_SIDE_CW = {side: SIDES[i % len(SIDES)] for i, side in enumerate(SIDES, start=1)}
OPPOSITE_SIDE = {side: SIDES[i % len(SIDES)] for i, side in enumerate(SIDES, start=2)}

# comparison funcs used to check that rects' adjacent sides are not past each
# other and therefor unable to be touching/bordering
ADJACENT_CMP = {
    'top': operator.gt,
    'right': operator.lt,
    'bottom': operator.lt,
    'left': operator.gt,
}

# "clock-wise" lines
SIDELINES_CW = {
    'top': ('topleft', 'topright'),
    'right': ('topright', 'bottomright'),
    'bottom': ('bottomright', 'bottomleft'),
    'left': ('bottomleft', 'topleft'),
}

MOVEKEYS = [
    pygame.K_UP,
    pygame.K_RIGHT,
    pygame.K_DOWN,
    pygame.K_LEFT,
]

MOVE_UP, MOVE_RIGHT, MOVE_DOWN, MOVE_LEFT = MOVEKEYS

MOVE_VELOCITY = {
    MOVE_UP: (0, -1),
    MOVE_DOWN: (0, 1),
    MOVE_RIGHT: (1, 0),
    MOVE_LEFT: (-1, 0),
}

get_sides = operator.attrgetter(*SIDES)

class TestIsBordering(unittest.TestCase):
    """
    Test functions for testing if rects are exactly bordering.
    """

    def setUp(self):
        self.r1 = pygame.Rect(0,0,10,10)
        self.r2 = self.r1.copy()

    def check_side(self, side):
        #
        oppo = OPPOSITE_SIDE[side]
        adj_cw = ADJACENT_SIDE_CW[side]
        adj_ccw = ADJACENT_SIDE_CCW[side]
        setattr(self.r2, oppo, getattr(self.r1, side))
        self.assertTrue(is_bordering(side, self.r1, self.r2))
        # simple reversal is touching
        self.assertTrue(is_bordering(oppo, self.r2, self.r1))
        # other sides are not touching
        self.assertFalse(is_bordering(adj_cw, self.r1, self.r2))
        self.assertFalse(is_bordering(adj_ccw, self.r1, self.r2))
        self.assertFalse(is_bordering(oppo, self.r1, self.r2))

    def test_bordering_right(self):
        self.check_side('right')

    def test_bordering_bottom(self):
        self.check_side('bottom')

    def test_bordering_left(self):
        self.check_side('left')

    def test_bordering_top(self):
        self.check_side('top')


class Snake:

    def __init__(self, body):
        """
        :param body: iterable of rects.
        """
        self.body = list(body)
        sizes = set(rect.size for rect in self.body)
        assert len(sizes) == 1
        self.size = first(sizes)

    @property
    def head(self):
        return self.body[-1]

    @property
    def tail(self):
        return self.body[0]

    @property
    def velocity(self):
        neck, head = self.body[-2:]
        hx, hy = head.center
        nx, ny = neck.center
        vx = (hx - nx) // self.size[0]
        vy = (hy - ny) // self.size[1]
        return (vx, vy)

    def slither(self, head_move, wrap=None):
        """
        Move the snake body in some direction.
        """
        for r1, r2 in zip(self.body, self.body[1:]):
            r1.center = r2.center
        # move head
        dx, dy = head_move
        head = self.head
        head.x += dx * self.size[0]
        head.y += dy * self.size[1]
        if wrap:
            head.x %= wrap.width
            head.y %= wrap.height


def unpack_line(line):
    "((a,b),(c,d)) -> (a,b,c,d)"
    p1, p2 = line
    a, b = p1
    c, d = p2
    return (a, b, c, d)

def intersects(line1, line2):
    """
    Return bool lines intersect.
    """
    # https://stackoverflow.com/a/24392281/2680592
    a, b, c, d = unpack_line(line1)
    p, q, r, s = unpack_line(line2)
    det = (c - a) * (s - q) - (r - p) * (d - b)
    if det == 0:
        return False
    lambda_ = ((s - q) * (r - a) + (p - r) * (s - b)) / det
    gamma = ((b - d) * (r - a) + (c - a) * (s - b)) / det
    result = (0 < lambda_ < 1) and (0 < gamma < 1)
    return result

def get_sides_dict(rect):
    return dict(zip(SIDES, get_sides(rect)))

def is_bordering(side, r1, r2):
    adj_ccw = ADJACENT_SIDE_CCW[side]
    adj_cw = ADJACENT_SIDE_CW[side]
    oppo = OPPOSITE_SIDE[side]
    adj_ccw_cmp = ADJACENT_CMP[adj_ccw]
    adj_cw_cmp = ADJACENT_CMP[adj_cw]
    return (
        getattr(r1, side) == getattr(r2, oppo)
        and not (
            # extremities are not past the other's opposite extremity
            # ex. for side == 'right',
            #     r1.top > r2.bottom or r1.bottom < r2.top
            adj_ccw_cmp(getattr(r1, adj_ccw), getattr(r2, adj_cw))
            or
            adj_cw_cmp(getattr(r1, adj_cw), getattr(r2, adj_ccw))
        )
    )

def is_bordering_any(r1, r2):
    return any(side for side in SIDES if is_bordering(side, r1, r2))

def chunk(pred, iterable):
    """
    Chunk items in iterable into runs.

    :param pred:
        Predicate receiving 2-tuple of the run so far and the current item.
    :param iterable:
        Some iterable.
    """
    items = iter(iterable)
    runs = []
    while True:
        run = []
        # consume one item or quit
        for item in items:
            run.append(item)
            break
        else:
            # nothing iterated, we're done
            break
        runs.append(run)

        for item in items:
            if pred(run, item):
                run.append(item)
            else:
                # "insert" back in the iterable so it gets chunked
                items = it.chain([item], items)
                break

    return runs

def get_sideline(rect, sidename):
    f = partial(getattr, rect)
    attrs = SIDELINES_CW[sidename]
    return tuple(map(f, attrs))

def get_sidelines(rect):
    f = partial(get_sideline, rect)
    return tuple(map(f, SIDELINES_CW))

def draw_snake(
    snake_body,
    window,
    color = 'red',
    width = 6,
):
    """
    Construct the lines from the centers of the rects, one to the next. Gather
    lines from all sides of all rects. Remove rect sides lines that intersect
    with any center lines. Draw the remaining lines.
    """
    def predicate(run, rect):
        # rect borders (exactly touches a side) with the last rect.
        return is_bordering_any(run[-1], rect)

    chunked_rects = chunk(predicate, snake_body)
    for rects in chunked_rects:
        rect_pairs = zip(rects, rects[1:])
        center_lines = [(r1.center, r2.center) for r1, r2 in rect_pairs]
        sides_lines = [line for rect in rects for line in get_sidelines(rect)]

        def intersects_any_center(line):
            intersects_center_line = partial(intersects, line)
            return any(map(intersects_center_line, center_lines))

        outline_lines = [line for line in sides_lines if not intersects_any_center(line)]
        for line in outline_lines:
            p1, p2 = line
            pygame.draw.line(window, color, p1, p2, width)

def first(iterable, pred=None, default=None):
    return next(filter(pred, iterable), default)

def render_lines(lines, font, surface, color, alignattrs, lastrect):
    """
    Render lines of text onto a surface with alignment from the last rect to
    the next.
    """
    toattr, fromattr = alignattrs
    for line in lines:
        image = font.render(line, True, color)
        kwargs = {toattr: getattr(lastrect, fromattr)}
        rect = image.get_rect(**kwargs)
        lastrect = surface.blit(image, rect)

def post_repaint():
    pygame.event.post(pygame.event.Event(REPAINT))

def run(
    snake,
    output_string = None,
    move_ms = None,
):
    """
    :param output_string:
        Optional new-style format string, taking one positional argument of
        frame number, used as path to write frames to.
        Example: path/to/frames/frame{:05d}.png
    :param move_ms: Move snake every milliseconds. Default: 250.
    """
    if move_ms is None:
        move_ms = 250

    clock = pygame.time.Clock()
    window = pygame.display.get_surface()
    frame = window.get_rect()
    gui_font = pygame.font.SysFont('monospace', int(min(frame.size)*.04))
    gui_frame = frame.inflate(*(-min(frame.size)*.1,)*2)
    background = window.copy()

    food = None

    def generate_food():
        nonlocal food
        length = min(snake.size)
        c, b, d, a = map(lambda x: x // length, get_sides(frame))
        # avoid placing off frame right- and bottom- side
        b -= 1
        d -= 1
        while True:
            i = random.randint(a, b)
            j = random.randint(c, d)
            x = i * snake.size[0]
            y = j * snake.size[1]
            food = pygame.Rect((x, y), snake.size)
            for body in snake.body:
                if body.colliderect(food):
                    break
            else:
                return

    # help text near middle-top
    render_lines(
        [
            'Outline Snake',
            'Hold space to animate',
            'D: toggles debug',
            'Escape: exit',
        ],
        gui_font,
        background,
        'ghostwhite',
        ('midtop', 'midbottom'), # align rect midtop to last rect's midbottom
        gui_frame.move(0, -gui_frame.height),
    )

    move_ms = move_ms or 250
    pygame.time.set_timer(MOVESNAKE, move_ms)
    pygame.time.set_timer(GENERATEFOOD, move_ms * 2)

    move_buffer = deque([MOVE_RIGHT], maxlen=15)
    frame_number = 0
    show_debug = False

    post_repaint()
    running = True
    while running:
        clock.tick(60)
        is_pressed = pygame.key.get_pressed()
        is_animating = is_pressed[pygame.K_SPACE]
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                elif event.key == pygame.K_d:
                    show_debug = not show_debug
                elif (
                    event.key in MOVEKEYS
                    and is_animating
                ):
                    move_buffer.append(event.key)
            elif (event.type == REPAINT):
                # draw
                window.blit(background, (0,)*2)
                # draw - snake body
                for rect in snake.body:
                    pygame.draw.rect(window, 'ghostwhite', rect)
                # draw - snake outline
                draw_snake(snake.body, window)
                # draw - food
                if food:
                    pygame.draw.rect(window, 'green', food, 0)
                if show_debug:
                    # draw - FPS
                    text_image = gui_font.render(f'{clock.get_fps():.2f}', True, 'ghostwhite')
                    window.blit(text_image, text_image.get_rect(topright = frame.topright))
                #
                pygame.display.flip()
                if is_animating and output_string:
                    path = output_string.format(frame_number)
                    pygame.image.save(window, path)
                    frame_number += 1
            if is_animating:
                if (event.type == GENERATEFOOD and food is None):
                    # new food
                    generate_food()
                    post_repaint()
                elif (event.type == MOVESNAKE):
                    # move snake
                    is_x, is_y = snake.velocity
                    want_move_key = move_buffer[-1]
                    want_x, want_y = MOVE_VELOCITY[want_move_key]
                    if (
                        (want_x and (want_x + is_x) != 0)
                        or
                        (want_y and (want_y + is_y) != 0)
                    ):
                        move = (want_x, want_y)
                    else:
                        move = (is_x, is_y)
                    snake.slither(move, wrap=frame)
                    post_repaint()
                    if food and snake.head.colliderect(food):
                        pygame.event.post(pygame.event.Event(EATFOOD))
                        break
                elif (event.type == EATFOOD):
                    # eat food
                    snake.body.insert(0, snake.tail.copy())
                    food = None
                    post_repaint()

def sizetype(string):
    """
    Parse string into a tuple of integers.
    """
    size = tuple(map(int, string.replace(',', ' ').split()))
    if len(size) == 1:
        size += size
    return size

def start(options):
    pygame.font.init()
    window = pygame.display.set_mode(options.size)
    frame = window.get_rect()

    side_length = options.side
    numbody = options.body
    body = [
        pygame.Rect(x * side_length, 0, side_length, side_length)
        for x in range(numbody)
    ]
    snake = Snake(body)

    run(
        snake,
        output_string = options.output,
        move_ms = options.movems,
    )

def cli():
    """
    Snake game demonstrating drawing an outline around the snake.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--size',
        default = '800,800',
        type = sizetype,
        help = 'Screen size. Default: %(default)s',
    )
    parser.add_argument(
        '--movems',
        type = int,
        help = 'Move snake every milliseconds.',
    )
    parser.add_argument(
        '--output',
        help = 'Format string for frame output.',
    )
    parser.add_argument(
        '--side',
        type = int,
        default = 20,
        help = 'Side length of body segments. Default: %(default)s',
    )
    parser.add_argument(
        '--body',
        type = int,
        default = 10,
        help = 'Length of snake. Default: %(default)s',
    )
    args = parser.parse_args()
    start(args)

if __name__ == '__main__':
    cli()
