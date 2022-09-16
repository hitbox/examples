import argparse
import contextlib
import itertools as it
import operator
import os

from collections import deque
from functools import partial
from types import SimpleNamespace

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    import pygame

class pressed_move:
    """
    Return tuple of scaled values from pressed keys.
    """

    def __init__(self, yneg, xpos, ypos, xneg):
        # arguments in clockwise order
        self.posneg_x = operator.itemgetter(xpos, xneg)
        self.posneg_y = operator.itemgetter(ypos, yneg)
        self.starmul = partial(it.starmap, operator.mul)

    def __call__(self, is_pressed, speed):
        speed_dir = [speed, -speed]
        # leaving a note because this is quick obscure
        # vel = sum(it.starmap(operator.mul, zip(arrow_rightleft(is_pressed),speed_dir)))
        vx = sum(self.starmul(zip(self.posneg_x(is_pressed),speed_dir)))
        vy = sum(self.starmul(zip(self.posneg_y(is_pressed),speed_dir)))
        return (vx, vy)


arrow_move = pressed_move(pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT)
wasd_move = pressed_move(pygame.K_w, pygame.K_d, pygame.K_s, pygame.K_a)

class Window:

    def __init__(self, title, frame, camera):
        self.title = title
        self.frame = frame
        self.camera = camera
        self.surf = pygame.Surface(self.camera.size, flags=pygame.SRCALPHA)

    def render(self, sprites):
        self.surf.fill('black')
        cx, cy = self.camera.topleft
        for sprite in sprites:
            x = sprite.rect.x - cx
            y = sprite.rect.y - cy
            rect = get_rect(sprite.rect, x=x, y=y)
            self.surf.blit(sprite.image, rect)
        return self.surf


def get_rect(*args, **kwargs):
    """
    :param *args:
        Optional rect used as base. Otherwise new (0,)*4 rect is created.
    :param kwargs:
        Keyword arguments to set on new rect.
    """
    if not len(args) in (0, 1):
        raise ValueError()
    if len(args) == 1:
        result = args[0].copy()
    else:
        result = pygame.Rect(0,0,0,0)
    for key, val in kwargs.items():
        setattr(result, key, val)
    return result

def simple_sprite(rect, color):
    image = pygame.Surface(rect.size)
    image.fill(color)
    return SimpleNamespace(
        image = image,
        rect = rect,
    )

def run(
    output_string = None,
):
    clock = pygame.time.Clock()
    fps = 60
    title_font = pygame.font.SysFont('monospace', 20)
    screen = pygame.display.get_surface()
    background = screen.copy()
    frame = screen.get_rect()

    help_images = [
        title_font.render('WASD - move window', True, 'ghostwhite'),
        title_font.render('Arrow - move green rect', True, 'ghostwhite'),
        title_font.render("Shift+Arrow - move window's camera", True, 'ghostwhite'),
    ]
    help_rects = [image.get_rect() for image in help_images]
    help_rects[0].topright = frame.inflate((-min(frame.size)*.2,)*2).topright
    for r1, r2 in zip(help_rects, help_rects[1:]):
        r2.topright = r1.bottomright
    for image, rect in zip(help_images, help_rects):
        background.blit(image, rect)

    windows = [
        Window(
            'Window Title',
            pygame.Rect(100,200,300,400),
            pygame.Rect(0,0,300,400),
        ),
    ]

    sprites = [
        simple_sprite(pygame.Rect(0,0,40,40), 'red'),
        simple_sprite(pygame.Rect(100,100,40,40), 'blue'),
        simple_sprite(pygame.Rect(100,300,40,40), 'green'),
    ]

    handlers = {}

    # should be in positive, negative order
    arrow_rightleft = operator.itemgetter(pygame.K_RIGHT, pygame.K_LEFT)
    arrow_downup = operator.itemgetter(pygame.K_DOWN, pygame.K_UP)

    wasd_rightleft = operator.itemgetter(pygame.K_d, pygame.K_a)
    wasd_downup = operator.itemgetter(pygame.K_s, pygame.K_w)

    speed = 4
    speed_dir = [speed, -speed]

    frame_num = 0
    frame_queue = deque()
    running = True
    while running:
        # tick and frame saving
        if output_string and frame_queue:
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
        # update
        is_pressed = pygame.key.get_pressed()

        if is_pressed[pygame.K_LSHIFT]:
            target = windows[-1].camera
        else:
            target = sprites[-1].rect

        vx, vy = arrow_move(is_pressed, speed)
        target.x += vx
        target.y += vy

        vx, vy = wasd_move(is_pressed, speed)
        windows[-1].frame.x += vx
        windows[-1].frame.y += vy

        # draw
        screen.blit(background, (0,)*2)
        # draw - windows contents
        for window in windows:
            surface = window.render(sprites)
            screen.blit(surface, window.frame)
        # draw - window decorations
        for window in windows:
            pygame.draw.rect(screen, 'ghostwhite', window.frame, 1)
            title_image = title_font.render(window.title, True, 'ghostwhite')
            title_rect = title_image.get_rect(bottomleft=window.frame.topleft)
            # title background
            title_frame = get_rect(title_rect, width=window.frame.width)
            pygame.draw.rect(screen, 'black', title_frame, 0)
            pygame.draw.rect(screen, 'ghostwhite', title_frame, 1)
            # title text
            screen.blit(title_image, title_rect)
        pygame.display.flip()
        if output_string:
            frame_queue.append(screen.copy())

def start(options):
    """
    Initialize and start run loop. Bridge between options and main loop.
    """
    pygame.font.init()
    pygame.display.set_caption('pygame - inventory')
    pygame.display.set_mode(options.size)
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
    # [X] Like Resident Evil 4 in 2d
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
