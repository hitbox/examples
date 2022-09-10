import argparse
import contextlib
import itertools as it
import operator
import os
import random
import unittest

from collections import deque
from functools import partial
from operator import attrgetter

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
        assert len(self.body) > 1
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

def how_bordering(r1, r2):
    for side in SIDES:
        if is_bordering(side, r1, r2):
            return side

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

def nwise(iterable, n=2, fill=None):
    "Take from iterable in `n`-wise tuples."
    iterables = it.tee(iterable, n)
    # advance iterables
    for offset, iterable in enumerate(iterables):
        # advance with for-loop to avoid catching StopIteration manually.
        for _ in zip(range(offset), iterable):
            pass
    return it.zip_longest(*iterables, fillvalue=fill)

def get_sideline(rect, sidename):
    attrs = SIDELINES_CW[sidename]
    return tuple(map(partial(getattr, rect), attrs))

def get_sidelines(rect):
    f = partial(get_sideline, rect)
    return tuple(map(f, SIDELINES_CW))

def outline_rects(rects):
    """
    Yield lines that outline the rects.
    """
    def by_touching_previous(run, rect):
        # rect borders (exactly touches a side) with the last rect.
        return is_bordering_any(run[-1], rect)

    for chunked_rects in chunk(by_touching_previous, rects):
        triples = nwise([None] + chunked_rects, 3)
        for triplet in triples:
            first, second, third = triplet
            if second is None and third is None:
                # avoid adding (head, None, None)
                break

            center_lines = [
                (r1.center, r2.center) for r1, r2 in nwise(triplet)
                if r1 and r2
            ]

            sides_rect = second or first
            outline = [
                line
                for rect in triplet
                for line in get_sidelines(sides_rect)
                if not any(intersects(line, cline) for cline in center_lines)
            ]

            yield (sides_rect, outline)

def draw_snake(
    snake_body,
    window,
    color = ('aquamarine', 'darkorchid'),
    background = ('honeydew', 'brown1'),
    width = 'auto',
):
    if width == 'auto':
        width = min(first(snake_body).size) // 4

    color1, color2 = map(pygame.Color, color)
    background1, background2 = map(pygame.Color, background)

    items = list(outline_rects(snake_body))
    for index, (rect, outline) in enumerate(items):
        percent = index / len(items)
        pygame.draw.rect(window, background1.lerp(background2, percent), rect)
        for p1, p2 in outline:
            pygame.draw.line(window, color1.lerp(color2, percent), p1, p2, width)

def validate_move(current, want):
    is_x, is_y = current
    want_x, want_y = want
    return (
        (want_x and (want_x + is_x) != 0)
        or
        (want_y and (want_y + is_y) != 0)
    )

def first(iterable, pred=None, default=None):
    return next(filter(pred, iterable), default)

def render_text_lines(lines, font, surface, color, alignattrs, lastrect):
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

def post_eatfood():
    pygame.event.post(pygame.event.Event(EATFOOD))

def render_arrow(size, color):
    arrow_up = pygame.Surface(size, flags=pygame.SRCALPHA)
    rect = arrow_up.get_rect()
    rect2 = rect.inflate(-rect.width // 2, 0)
    rect2.height //= 3
    pygame.draw.polygon(
        arrow_up,
        color,
        [
            rect.midtop,
            rect2.bottomright,
            rect2.bottomleft
        ]
    )
    rect3 = rect2.inflate(-rect.width // 3, 0)
    rect3.height = rect.bottom - rect2.bottom
    rect3.top = rect2.bottom
    pygame.draw.rect(arrow_up, color, rect3)
    return arrow_up

def run(
    snake,
    output_string = None,
    move_ms = None,
    segments_per_food = None,
):
    """
    :param output_string:
        Optional new-style format string, taking one positional argument of
        frame number, used as path to write frames to.
        Example: path/to/frames/frame{:05d}.png
    :param move_ms: Move snake every milliseconds. Default: 250.
    """
    if move_ms is None:
        move_ms = 100

    if segments_per_food is None:
        segments_per_food = 1

    clock = pygame.time.Clock()
    window = pygame.display.get_surface()
    frame = window.get_rect()
    gui_font = pygame.font.SysFont('monospace', int(min(frame.size)*.04))
    gui_frame = frame.inflate(*(-min(frame.size)*.1,)*2)
    background = window.copy()

    arrow_up = render_arrow(snake.size, 'ghostwhite')
    arrow_right = pygame.transform.rotate(arrow_up, -90)
    arrow_down = pygame.transform.rotate(arrow_right, -90)
    arrow_left = pygame.transform.rotate(arrow_down, -90)

    arrow_from_key = {
        pygame.K_UP: arrow_up,
        pygame.K_RIGHT: arrow_right,
        pygame.K_DOWN: arrow_down,
        pygame.K_LEFT: arrow_left,
    }

    # help text near middle-top
    render_text_lines(
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

    def draw():
        nonlocal frame_number
        window.blit(background, (0,)*2)

        # snake
        draw_snake(snake.body, window)
        # food
        if food:
            pygame.draw.rect(window, 'green', food, 0)
        if show_debug:
            # FPS
            text_image = gui_font.render(f'{clock.get_fps():.2f}', True, 'ghostwhite')
            window.blit(text_image, text_image.get_rect(topright = frame.topright))
        #
        if False:
            width = snake.size[0]
            image = pygame.Surface((width*len(want_keys), snake.size[1]))
            positions = range(0, len(want_keys)*width, width)
            for x, key in zip(positions, want_keys):
                arrow_image = arrow_from_key[key]
                image.blit(arrow_image, (x, 0))
            window.blit(image, image.get_rect(midright=gui_frame.midright))

        pygame.display.flip()
        if output_string:
            # save frame to output
            path = output_string.format(frame_number)
            pygame.image.save(window, path)
            frame_number += 1

    def handle(event):
        nonlocal show_debug
        nonlocal food
        if event.type == pygame.VIDEOEXPOSE:
            post_repaint()
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                # stop animating
                pygame.time.set_timer(MOVESNAKE, 0)
                pygame.time.set_timer(GENERATEFOOD, 0)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # quit
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            elif event.key == pygame.K_d:
                # toggle debug
                show_debug = not show_debug
            elif event.key == pygame.K_SPACE:
                # begin animating
                pygame.time.set_timer(MOVESNAKE, move_ms)
            elif event.key in MOVEKEYS:
                # buffer input
                want_keys.append(event.key)
        elif (event.type == REPAINT):
            # repaint screen
            draw()
        elif (event.type == GENERATEFOOD):
            # new food
            generate_food()
            post_repaint()
            pygame.time.set_timer(GENERATEFOOD, 0)
        elif (event.type == MOVESNAKE):
            # move snake
            want_key = want_keys[-1]
            move = snake.velocity
            want = MOVE_VELOCITY[want_key]
            if validate_move(move, want):
                move = want
            snake.slither(move, wrap=frame)
            if food and snake.head.colliderect(food):
                # feed snake
                post_eatfood()
            post_repaint()
        elif (event.type == EATFOOD):
            # eat food
            food = None
            for _ in range(segments_per_food):
                snake.body.insert(0, snake.tail.copy())
            pygame.time.set_timer(GENERATEFOOD, move_ms * 4)
            post_repaint()

    frame_number = 0
    show_debug = False
    want_keys = deque([pygame.K_RIGHT], maxlen=1)

    pygame.time.set_timer(GENERATEFOOD, move_ms * 4)
    running = True
    while running:
        clock.tick(60)
        is_pressed = pygame.key.get_pressed()
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                handle(event)

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
        segments_per_food = options.food,
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
    parser.add_argument(
        '--food',
        type = int,
        help = 'Segments to add per food ate. Default: %(default)s',
    )
    args = parser.parse_args()
    start(args)

if __name__ == '__main__':
    cli()
