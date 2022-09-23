import argparse
import contextlib
import itertools as it
import os
import unittest

from collections import defaultdict
from collections import deque
from functools import partial
from functools import singledispatch
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

SIDES = ('top', 'right', 'bottom', 'left')

OPPOSITE_SIDE = {side: SIDES[i % len(SIDES)] for i, side in enumerate(SIDES, start=2)}

ADJACENT_NAMES = {
    'top': ('left', 'right'),
    'right': ('top', 'bottom'),
    'bottom': ('right', 'left'),
    'left': ('bottom', 'top'),
}

# "clock-wise" lines
SIDELINES_CW = dict((name, ADJACENT_NAMES[name]) for name in SIDES)

DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# delta direction vector back to side name
DELTAS_NAMES = dict(zip(DELTAS, SIDES))

class TestModOffset(unittest.TestCase):
    """
    Test modulo with offset.
    """

    def test_modoffset(self):
        c = 3
        n = 8
        self.assertEqual(modo(3, n, c), 3)
        self.assertEqual(modo(8, n, c), 3)
        self.assertEqual(modo(9, n, c), 4)
        self.assertEqual(modo(0, n, c), 5)


class TestLerp(unittest.TestCase):
    """
    Test linear interpolation.
    """

    def check_ends(self, a, b):
        self.assertEqual(lerp(a,b,0), a)
        self.assertEqual(lerp(a,b,1), b)

    def test_lerp(self):
        a, b = 0, 1
        self.check_ends(a, b)
        self.assertEqual(lerp(a,b,.5), .5)

    def test_lerp_tuple(self):
        a = (0, 0, 0)
        b = (1, 1, 1)
        self.check_ends(a, b)
        self.assertEqual(lerp(a,b,.5), (.5,)*3)

    def test_lerp_pygame_color(self):
        a = pygame.Color('black')
        b = pygame.Color('white')
        self.check_ends(a, b)


class TestInvLerp(unittest.TestCase):
    """
    Test inverse linear interpolation.
    """

    def check_ends(self, a, b):
        self.assertEqual(invlerp(a, b, 0), a)
        self.assertEqual(invlerp(a, b, 1), b)

    def test_invlerp(self):
        a, b = 0, 1
        self.check_ends(a, b)

    def test_invlerp_tuple(self):
        a = (0, 0, 0)
        b = (1, 1, 1)
        self.check_ends(a, b)

    def test_lerp_pygame_color(self):
        a = pygame.Color('black')
        b = pygame.Color('white')
        self.check_ends(a, b)


class TestRemap(unittest.TestCase):
    """
    Test remap ranges.
    """

    def check_ends(self, a, b):
        self.assertEqual(remap(a, b, 0), b[0])
        self.assertEqual(remap(a, b, 1), b[1])

    def test_remap(self):
        a, b = (0, 1), (0, 2)
        self.check_ends(a, b)


class Cursor:
    """
    Cursor selector on grid.
    """

    def __init__(self, rect, holding=None, hovering=None):
        self.rect = rect
        self.holding = holding
        self.hovering = hovering

    def update_hovering(self, items):
        for item in items:
            if self.rect.colliderect(wrap(item.body)):
                self.hovering = item
                break
        else:
            self.hovering = None


class AttributeAnimation:
    """
    Set an object's attribute from the next value of an iterable, every call.
    """

    def __init__(self, target, attr, iterable):
        self.target = target
        self.attr = attr
        self.iterable = iterable

    def __call__(self):
        value = next(self.iterable)
        setattr(self.target, self.attr, value)


class CursorRenderer:
    """
    Render cursor when it is not holding an item.
    """

    def __init__(self, items):
        self.items = items

    def __call__(self, surf, cursor):
        if cursor.holding:
            return
        # not holding
        if cursor.hovering:
            # is hovering
            cursor_rect = wrap(cursor.hovering.body)
        else:
            # not hovering
            cursor_rect = cursor.rect
        if cursor.hovering:
            # shimmer hovering item
            image = pygame.Surface(cursor_rect.size, flags=pygame.SRCALPHA)
            image.fill(cursor.fill_color)
            surf.blit(image, cursor_rect)
        pygame.draw.rect(surf, cursor.color, cursor_rect, 1)


class InventoryItemRenderer:
    """
    Render items in an order that keeps the held item on top.
    """

    def __init__(self, shadow_color=None):
        if shadow_color is None:
            shadow_color = get_color('green', a=255//2)
        self.shadow_color = shadow_color

    def __call__(self, surf, grid, cursor):
        """
        Render all inventory items onto grid.
        """
        def draw_body(item, body_rect):
            surf.blit(render_rect(body_rect, item.border, item.color), body_rect)

        for item in grid.items:
            if item is cursor.holding:
                continue
            draw_body(item, wrap(item.body))

        if cursor.holding:
            body_rect = wrap(cursor.holding.body)
            # draw drop shadow on top
            surf.blit(render_rect(body_rect, fill_color=self.shadow_color), body_rect)
            # draw held item to appear above the grid
            inflate_size = (min(grid.rect.size)*.05, ) * 2
            elevated_rect = grid.rect.inflate(inflate_size)
            elevated_rect.midbottom = (grid.rect.centerx, grid.rect.bottom - 4)
            # remap from grid to elevated rect
            body_rect.center = remap(grid.rect, elevated_rect, body_rect.center)
            draw_body(cursor.holding, body_rect)


class Grid:
    """
    Attributes of the grid to pack items into.
    """

    def __init__(self, rows, cols, cell_size):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

    @property
    def image_size(self):
        return (
            self.cols * self.cell_size[0],
            self.rows * self.cell_size[1]
        )


class EventHandlers:

    def __init__(self):
        self.handlers = defaultdict(list)

    def __call__(self, event_type):
        def decorator(func):
            self.handlers[event_type].append(func)
            return func
        return decorator

    def for_event(self, event):
        return self.handlers[event.type]


eventhandler = EventHandlers()

get_bounds = attrgetter(*SIDES)

def resolveattr(attr, obj, data, fallback):
    return getattr(obj, attr, data.get(attr, fallback))

def render_lines(font, color, lines, antialias=True):
    """
    Render lines of text with a font.
    """
    def render_line(line):
        "Reorganize arguments for use with map"
        return font.render(line, antialias, color)
    return list(map(render_line, lines))

def modo(a, b, c):
    """
    modulo offset, returns a % b shifted for offset c.
    """
    return c + ((a - c) % (b - c))

# XXX:
# - singledispatch only considers the first argument?
# - is this a sensible thing to do?

@singledispatch
def lerp(a, b, t):
    "Return position between a and b at 'time' t"
    return a * (1 - t) + b * t

@lerp.register
def lerp_tuple(a:tuple, b, t):
    return tuple(lerp(i1, i2, t) for i1, i2 in zip(a, b))

@lerp.register
def lerp_pygame_color(a:pygame.Color, b, t):
    return a.lerp(b, t)

@singledispatch
def invlerp(a, b, x):
    "Return 'time' from position x between a and b."
    return (x - a) / (b - a)

@invlerp.register
def invlerp_tuple(a:tuple, b, x):
    return tuple(invlerp(*args) for args in zip(a, b, it.repeat(x)))

@invlerp.register
def invlerp_pygame_color(a:pygame.Color, b, x):
    # could not find that pygame has inverse lerp for colors, have to roll our own
    return pygame.Color(lerp(tuple(a), tuple(b), x))

@singledispatch
def remap(r1:[int,float], r2, x):
    "position x mapped from range a-b to range c-d."
    a, b = r1
    c, d = r2
    return x*(d-c)/(b-a) + c-a*(d-c)/(b-a)

@remap.register
def remap_rect(rect1:pygame.Rect, rect2, p):
    "remap x,y position from rect1 to a position in rect2"
    x, y = p
    r1x = (rect1.left, rect1.right)
    r2x = (rect2.left, rect2.right)
    r1y = (rect1.top, rect1.bottom)
    r2y = (rect2.top, rect2.bottom)
    return (remap(r1x, r2x, x), remap(r1y, r2y, y))

def post_movecursor(delta, grid):
    event = pygame.event.Event(MOVECURSOR, delta=delta)
    pygame.event.post(event)

def post_grab(cursor, items):
    event = pygame.event.Event(GRAB, cursor=cursor, items=items)
    pygame.event.post(event)

def post_drop(cursor, items):
    event = pygame.event.Event(DROP, cursor=cursor, items=items)
    pygame.event.post(event)

def post_rotate_holding(cursor):
    event = pygame.event.Event(ROTATE_HOLDING, cursor=cursor)
    pygame.event.post(event)

@eventhandler(pygame.KEYDOWN)
def on_keydown(event):
    if event.key in (pygame.K_q, ):
        # quit
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    elif event.key in MOVEKEY_DELTA:
        # key to move event
        delta = MOVEKEY_DELTA[event.key]
        post_movecursor(delta, event.grid)
    elif event.key == pygame.K_TAB:
        # rotate item cursor is holding
        post_rotate_holding(event.cursor)
    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
        # grab/drop item
        if event.cursor.holding:
            post_drop(event.cursor, event.grid.items)
        else:
            post_grab(event.cursor, event.grid.items)

@eventhandler(MOVECURSOR)
def on_movecursor(event):
    """
    Move everything as one to keep cursor relative position
    """
    move_cursor(event.cursor, event.delta, event.grid.items, event.grid.rect)

@eventhandler(ROTATE_HOLDING)
def on_rotate(event):
    """
    Rotate item current held by cursor.
    """
    if event.cursor.holding:
        rotate_holding(event.cursor, event.grid.rect)

@eventhandler(GRAB)
def on_grab(event):
    item = cursor_collideitem(event.cursor, event.grid.items)
    if item:
        event.cursor.holding = item

@eventhandler(DROP)
def on_drop(event):
    was_holding = event.cursor.holding
    other_colliding = [
        item
        for item in cursor_collideitem_all(event.cursor, event.grid.items)
        if item is not was_holding
    ]
    if len(other_colliding) == 1:
        # item dropped onto another, pick the other up
        event.cursor.holding = other_colliding[0]
        event.cursor.rect.clamp_ip(wrap(other_colliding[0].body))
    elif len(other_colliding) == 0:
        # item dropped into empty space
        event.cursor.holding = None

def nwise(iterable, n=2, fill=None):
    "Take from iterable in `n`-wise tuples."
    iterables = it.tee(iterable, n)
    # advance iterables for offsets
    for offset, iterable in enumerate(iterables):
        # advance with for-loop to avoid catching StopIteration manually.
        for _ in zip(range(offset), iterable):
            pass
    return it.zip_longest(*iterables, fillvalue=fill)

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

def wrap(rects):
    """
    Wrap iterable of rects in a bounding box.
    """
    boundaries = zip(*map(get_bounds, rects))
    tops, rights, bottoms, lefts = boundaries
    top = min(tops)
    right = max(rights)
    bottom = max(bottoms)
    left = min(lefts)
    width = right - left
    height = bottom - top
    return pygame.Rect(left, top, width, height)

def move_as_one(rects, **kwargs):
    """
    Move iterable of rects as if they were one, to destination provided in kwargs.
    """
    rects1, rects2 = it.tee(rects)
    original = wrap(rects1)
    destination = get_rect(original, **kwargs)
    dx = destination.x - original.x
    dy = destination.y - original.y
    for rect in rects2:
        rect.x += dx
        rect.y += dy
    return destination

def align(rects, attrmap):
    """
    Align rects in iterable order
    :param rects: list of rects.
    :param attrmap: mapping of attribute names.
    """
    # allow for many ways to give mapping but this uses dict interface
    attrmap = dict(attrmap)
    for prevrect, rect in nwise(rects):
        if not rect:
            continue
        for key, prevkey in attrmap.items():
            setattr(rect, key, getattr(prevrect, prevkey))

def get_color(arg, **kwargs):
    """
    Set attributes on a color while instantiating it.
    Example:

    get_color('red', a=255//2)
    For getting 'red' and setting the alpha at the same time.
    """
    color = pygame.Color(arg)
    for key, val in kwargs.items():
        setattr(color, key, val)
    return color

def cursor_collideitem(cursor, items):
    if cursor.holding:
        cursor_rect = wrap(cursor.holding.body)
    else:
        cursor_rect = cursor.rect
    for item in items:
        if cursor_rect.colliderect(wrap(item.body)):
            return item

def cursor_collideitem_all(cursor, items):
    if cursor.holding:
        cursor_rect = wrap(cursor.holding.body)
    else:
        cursor_rect = cursor.rect
    return [item for item in items if cursor_rect.colliderect(wrap(item.body))]

def makegrid(rect_size, rows, columns):
    """
    Return list of rects of given size, arranged in a grid of rows and columns.
    """
    width, height = rect_size
    yield from (
        pygame.Rect(width*i, height*j, width, height)
        for i, j in it.product(range(rows), range(columns))
    )

def render_rect(
    rect,
    border_color = None,
    fill_color = None,
    width = 1,
    flags = pygame.SRCALPHA
):
    # allows "rect drawing" with alpha in color
    surf = pygame.Surface(rect.size, flags=flags)
    if fill_color:
        surf.fill(fill_color)
    if border_color:
        pygame.draw.rect(surf, border_color, rect, width)
    return surf

def draw_grid(surface, cell_size, *colors, line_width=1):
    if len(colors) != line_width:
        raise ValueError
    cell_width, cell_height = cell_size
    rect = surface.get_rect()
    offset_start = -line_width // 2
    for x in range(cell_width, rect.width, cell_width):
        for offset, color in enumerate(colors, start=offset_start):
            p1 = (x+offset, 0)
            p2 = (x+offset, rect.height)
            pygame.draw.line(surface, color, p1, p2, 1)
    for y in range(cell_height, rect.height, cell_height):
        for offset, color in enumerate(colors, start=offset_start):
            p1 = (0, y+offset)
            p2 = (rect.width, y+offset)
            pygame.draw.line(surface, color, p1, p2, 1)

def draw_cells(surface, rect, color, width=1):
    """
    Draw the grid the way the cells are displayed--next to each other.
    """
    frame = surface.get_rect()
    # save x
    x = rect.x
    while True:
        while True:
            pygame.draw.rect(surface, color, rect, width)
            rect = get_rect(rect, left=rect.right)
            if not frame.contains(rect):
                break
        rect = get_rect(rect, x=x, top=rect.bottom)
        if not frame.contains(rect):
            break

def move_cursor(cursor, delta, items, grid_rect):
    """
    Move inventory cursor. Moves wrap around grid_rect. Moves take whatever
    cursor is holding with it. Cursor warps through items.
    """
    # TODO
    # - moving left, jump across object that causes wrap, makes cursor appear
    #   to move right instead
    dx, dy = delta
    if cursor.holding:
        # Move as if from the topleft of the holding item's body rects. The
        # cursor could be anywhere inside the thing it's holding.
        x, y = cursor.holding.body[0].topleft
    else:
        x, y = cursor.rect.topleft
    x += dx * cursor.rect.width
    y += dy * cursor.rect.height

    # fixup x, y as we go along
    if not cursor.holding:
        if cursor.hovering:
            # warping through items when not holding something
            item_rect = wrap(cursor.hovering.body)
            if dx < 0:
                x = item_rect.left - cursor.rect.width
            elif dx > 0:
                x = item_rect.right
            if dy < 0:
                y = item_rect.top - cursor.rect.height
            elif dy > 0:
                y = item_rect.bottom
            # TODO
            # - do a clamp on (x,y) here to prevent wrapping when jumping
            #   through item

    # gather up rects to move and adjust the right+bottom values for wrapping
    # this is necessary because the wrapping occurs against the topleft
    right, bottom = grid_rect.bottomright
    rects = [cursor.rect]
    if cursor.holding:
        # wrap for dimensions of item, not cursor
        width, height = wrap(cursor.holding.body).size
        right -= width - cursor.rect.width
        bottom -= height - cursor.rect.height
        # add what cursor is holding to list
        rects.extend(cursor.holding.body)

    x = modo(x, right, grid_rect.left)
    y = modo(y, bottom, grid_rect.top)
    move_as_one(rects, x=x, y=y)

    cursor.update_hovering(items)

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
    table = ( list(items) for key, items in grouped )
    #
    rotated_table = zip(*table)
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

def clamp_many(rects, inside):
    # XXX: is move_as_one doing too much? since we're duplicating stuff here.
    wrapped = wrap(rects)
    moved = wrapped.clamp(inside)
    move_as_one(rects, topleft=moved.topleft)

def rotate_holding(cursor, grid_rect):
    """
    Rotate the item the cursor is holding and constrain to the grid.
    """
    rects = cursor.holding.body + [cursor.rect]
    # TODO
    # - rotate can put item outside grid
    rotate_rects(rects)
    clamp_many(rects, grid_rect)
    cursor.rect.clamp_ip(wrap(cursor.holding.body))

def place_item(item, grid):
    """
    """
    item_wrap = wrap(item.body)
    cells = list(makegrid(grid.cell_size, grid.rows, grid.cols))
    move_as_one(cells, topleft=grid.rect.topleft)
    filled = [rect for item in grid.items for rect in item.body]
    for cell_rect in cells:
        item_wrap = get_rect(item_wrap, topleft=cell_rect.topleft)
        if not any(item_wrap.colliderect(filled_rect) for filled_rect in filled):
            return move_as_one(item.body, topleft=cell_rect.topleft)

def item_body_area(item):
    """
    Area of an items entire body of rects.
    """
    wrapped = wrap(item.body)
    return wrapped.width * wrapped.height

def devel_items(cell_size):
    """
    Hard-coded item objects for development.
    """
    pistol = SimpleNamespace(
        name = 'Pistol',
        color = 'red',
        border = 'darkred',
        rows = 2,
        cols = 3,
        font = SimpleNamespace(
            color = 'red',
        ),
    )
    pistol.body = list(makegrid(cell_size, pistol.rows, pistol.cols))
    rotate_rects(pistol.body)

    rifle = SimpleNamespace(
        name = 'Rifle',
        color = 'burlywood4',
        border = 'burlywood',
        rows = 1,
        cols = 9,
        font = SimpleNamespace(
            color = 'burlywood4',
        ),
    )
    rifle.body = list(makegrid(cell_size, rifle.rows, rifle.cols))
    rotate_rects(rifle.body)

    grenade = SimpleNamespace(
        name = 'Grenade',
        color = 'darkgreen',
        border = 'green',
        rows = 2,
        cols = 1,
        font = SimpleNamespace(
            color = 'darkgreen',
        ),
    )
    grenade.body = list(makegrid(cell_size, grenade.rows, grenade.cols))
    rotate_rects(grenade.body)

    chicken_egg = SimpleNamespace(
        name = 'Egg',
        color = 'oldlace',
        border = 'ghostwhite',
        rows = 1,
        cols = 1,
        font = SimpleNamespace(
            color = 'oldlace',
        ),
    )
    chicken_egg.body = list(makegrid(cell_size, chicken_egg.rows, chicken_egg.cols))
    rotate_rects(chicken_egg.body)

    items = [pistol, rifle, grenade, chicken_egg]
    return items

def run(settings):
    # help text
    help_ = SimpleNamespace(
        color = 'ghostwhite',
        normal_font = pygame.font.SysFont('nope', 32),
        frame = settings.frame.inflate((-min(settings.frame.size)//32, ) * 2),
        lines = [
            'Arrow keys to move',
            'Return or Space to grab and drop',
            'Tab to rotate',
        ],
    )
    # render text images and rects
    help_.images = render_lines(help_.normal_font, help_.color, help_.lines)
    help_.rects = [image.get_rect() for image in help_.images]
    # align text rects
    help_.rects[0].topright = help_.frame.topright
    align(help_.rects, {'topright': 'bottomright'})

    # grid
    grid = Grid(rows=7, cols=11, cell_size=(60,)*2)
    grid.image = pygame.Surface(grid.image_size, flags=pygame.SRCALPHA)
    draw_cells(grid.image, pygame.Rect((0,)*2, grid.cell_size), 'azure4')
    grid.rect = grid.image.get_rect(center=settings.frame.center)
    # make and move grid cells; and sort for placement later
    grid.items = sorted(devel_items(grid.cell_size), key=item_body_area)
    for item in grid.items:
        move_as_one(item.body, topleft=grid.rect.topleft)
    # place items on grid
    items = grid.items[:]
    while items:
        item = items.pop()
        place_item(item, grid)

    # back unchanging images into background
    for image, rect in zip(help_.images, help_.rects):
        settings.background.blit(image, rect)
    settings.background.blit(grid.image, grid.rect)

    cursor = Cursor(rect=pygame.Rect(grid.rect.topleft, grid.cell_size))
    cursor.color = pygame.Color('yellow')
    cursor.fill_color = pygame.Color('yellow')
    cursor.renderer = CursorRenderer(grid.items)

    item_renderer = InventoryItemRenderer()

    animations = [
        AttributeAnimation(
            cursor.fill_color,
            'a', # alpha
            it.cycle(
                int(lerp(255*.25, 255*.50, frametime/10))
                for time in it.chain(range(10), range(9, 0, -1))
                for frametime in it.repeat(time, 2)
            )
        ),
    ]

    frame_queue = deque()
    frame_num = 0

    def save_frame():
        """
        Save a frame from the frame queue.
        """
        nonlocal frame_num
        frame_image = frame_queue.popleft()
        path = settings.output_string.format(frame_num)
        pygame.image.save(frame_image, path)
        frame_num += 1

    running = True
    while running:
        # tick and frame saving
        if settings.output_string and frame_queue:
            # save frames until exhausted or fps is achieved
            while frame_queue and settings.clock.get_fps() > fps:
                save_frame()
        elapsed = settings.clock.tick(settings.fps)
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # stop main loop after this frame
                running = False
            else:
                event.cursor = cursor
                event.grid = grid
                for handler in eventhandler.for_event(event):
                    handler(event)
        # update
        for animation in animations:
            animation()
        # draw
        settings.window.blit(settings.background, (0,)*2)
        item_renderer(settings.window, grid, cursor)
        cursor.renderer(settings.window, cursor)
        # draw - item name
        item = cursor.hovering or cursor.holding
        if item:
            image = help_.normal_font.render(item.name, True, item.font.color)
            settings.window.blit(image, image.get_rect(bottomleft=help_.frame.bottomleft))
        # draw - finish
        pygame.display.flip()
        if settings.output_string:
            frame_queue.append(settings.window.copy())
    # consume remaining frame queue
    if settings.output_string and frame_queue:
        while frame_queue:
            save_frame()

def start(options):
    """
    Initialize and start run loop. Bridge between options and main loop.
    """
    pygame.font.init()
    pygame.display.set_caption('pygame - inventory')

    settings = SimpleNamespace(
        window = pygame.display.set_mode(options.size),
        clock = pygame.time.Clock(),
        fps = 60,
        output_string = options.output,
    )
    settings.background = settings.window.copy()
    settings.frame = settings.window.get_rect()

    run(settings)

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
    # [X] Like Resident Evil 4 in 2d
    # [X] grab / drop
    # [ ] Auto arrange with drag+drop animations
    # [X] Rotate items
    # [ ] Stacking?
    # [ ] Combine to new item?
    # [ ] Stealing minigame. Something chases or attacks your cursor.
    # [ ] Moving through items are a way of jumping, could be part of gameplay.
    # [ ] `.removable` attribute to prevent dropping.
    # [ ] Drop item grid to drop items.
    # [ ] Inform7, dialog (https://linusakesson.net/dialog/), etc. engine to do
    #     things like automatically unlock the unremovable item before dropping
    #     it, if have the key.
    # [ ] Another inventory for incorporeal items, like keys?
    # [ ] Inner grid games inside your inventory. Like tic-tac-toe, maybe even
    #     checkers or chess.
    # [ ] state machine for hovering -> grabbing -> dropping...
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
