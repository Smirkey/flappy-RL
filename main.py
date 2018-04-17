from flappyBird import *
import pygame
import cv2
import numpy as np

def main():
    score = 0
    pygame.display.init()
    Env = env(256, 256)
    screen = pygame.display.set_mode((Env.width, Env.height))
    action = [0,1]
    while 1:
        time.sleep(1/24)
        for event in pygame.event.get():
            if event.type == 2  and event.key == 32:
                action = [1,0]
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        pixels, reward, terminal = Env.step(action)
        if reward == 1:
            score +=1
            print(score)
        if terminal:
            score = 0
        surf = pygame.surfarray.make_surface(pixels)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        action = [0,1]
           
 
if __name__ == '__main__':
    main()
