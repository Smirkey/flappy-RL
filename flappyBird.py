from pygame.locals import *
import pygame
import random

class bird():

    def __init__(self, w, h):
        self.x = int(w / 2)
        self.y = int(h / 2)
        self.velocity = 0
        self.force = -0.018 * h
        self.radius = int(h / 50)
        self.alive = True
        self.score = 0

    def show(self, surf):
        pygame.draw.circle(surf, [255, 255, 255], [
                           int(self.x), int(self.y)], self.radius, 0)

    def up(self):
        self.velocity += self.force


def getNewPipe(h, w):
    return pipe(h, w)


def isBetween(pos, x1, x2):
    return pos > x1 and pos < x2


def isTouching(y, radius, y2, gap):
    if y > y2 + radius and y + radius < y2 + gap:
        return False
    else:
        return True


def createSurf(w, h):
    return pygame.Surface((w, h))


class pipe:

    def __init__(self, h, w):
        self.h = h
        self.width = int(self.h / 8)
        self.x = -5*self.width
        self.gap = self.h / 4 
        self.y = int(random.random() * (self.h - self.gap))
        self.speed = 0.0147058824 * w
        self.notCountedYet = True

    def show(self, surf):
        pygame.draw.rect(surf, [0, 255, 0], (self.x, self.y + self.gap,
                                             self.width, self.h - self.gap - self.y), 0)  # bas
        pygame.draw.rect(surf, [0, 255, 0], (self.x, 0,
                                             self.width, self.y), 0)  # haut

    def update(self):
        self.x += self.speed

    def hit(self, bird):
        if isBetween(bird.x - bird.radius, self.x, self.x + self.width):
            return isTouching(bird.y, bird.radius, self.y, self.gap)


class env():

    def __init__(self, w, h):
        self.framecount = 0
        self.width = w
        self.height = h
        self.bird = bird(self.width, self.height)
        self.pipes = [pipe(h, w)]
        self.gravity = 0.0015625 * h
        self.reward = 0
        self.surf = createSurf(w, h)

    def show(self, surf):
        for pipe in self.pipes:
            pipe.show(surf)
        self.bird.show(surf)

    def step(self, action):

        if not self.bird.alive:
            self.bird = self.bird = bird(self.width, self.height)
            self.pipes = [getNewPipe(self.height, self.width)]
            self.surf.fill((0, 0, 0))
            self.framecount = 0

        if action[0] == 1:
            self.bird.up()

        self.reward = 0.1  
        self.surf.fill((0, 0, 0))
        self.framecount += 1

        for pipe in self.pipes:
            pipe.show(self.surf)
            pipe.update()

            if pipe.x > self.width:
                self.pipes.remove(pipe)

            if pipe.hit(self.bird):
                self.bird.alive = False

            if pipe.notCountedYet:
                if pipe.x > self.bird.x + self.bird.radius and self.bird.alive:
                    self.reward = 1
                    pipe.notCountedYet = False

        if self.framecount % 50 == 0:
            self.pipes.append(getNewPipe(self.height, self.width))
            self.framecount = 0

        self.bird.velocity += self.gravity
        self.bird.velocity *= 0.97
        self.bird.y += self.bird.velocity
        self.bird.show(self.surf)

        if self.bird.y > self.height - self.bird.radius:
            self.bird.y = self.height - self.bird.radius
            self.bird.alive = False
            self.bird.velocity = 0

        if self.bird.y < self.bird.radius:
            self.bird.alive = False
            self.bird.y = self.bird.radius
            self.bird.velocity = 0

        if not self.bird.alive:
            self.reward = -1

        return pygame.surfarray.array3d(self.surf), self.reward, not self.bird.alive
