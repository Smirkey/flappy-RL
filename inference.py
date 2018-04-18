import flappyBird as game
import cv2
import numpy as np
import tensorflow as tf
import random
import os
import argparse
import pygame
import time
from train import model, version, ACTIONS, resizeAndDiscolor, WIDTH, HEIGHT

FRAME_PER_ACTION = 1
EPSILON = 0.001



def makeVideo(frames, score):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=False, default='games/{}/{}.mp4'.format(version, score), help="output video file")
    args = vars(ap.parse_args())
    output = args['output']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    width, height, channels = frames[0].shape
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    if not os.path.exists('./games/' + version):
                os.makedirs('./games/' + version)
    
    for frame in frames:
        out.write(frame)

    out.release()
    print("video out")

        
        
    
def infer(frame, logits, sess):

    saver = tf.train.Saver()
    do_nothing = np.zeros(ACTIONS)
    do_nothing[1] = 1

    Game = game.env(WIDTH, HEIGHT)
    f_t, r_0, terminal = Game.step(do_nothing)
    f_t = resizeAndDiscolor(f_t)
    s_t = np.stack((f_t, f_t, f_t, f_t), axis=2)
    
    
    
    epsilon = EPSILON
    t = 0
    sess.run(tf.global_variables_initializer())

    score = 0
    currentBest = 0
    frames = []

    checkpoint = tf.train.get_checkpoint_state("model/{}".format(version))
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("model successfully loaded:", checkpoint.model_checkpoint_path)

    while 1:
        a_t = np.zeros([ACTIONS])
        a_index = 1

        output = logits.eval(feed_dict = {frame: [s_t]})[0]
        
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                a_index = random.randrange(ACTIONS)
                a_t[a_index] = 1
            else:
                a_index = np.argmax(output)
                a_t[a_index] = 1
        else:
            a_t[a_index] = 1

        f_t1, r_t, terminal = Game.step(a_t)

        if r_t == 1:
            score += 1
            
        if terminal:
            print(score)
            if score > currentBest:
                print("new Record !")
                currentBest = score
                makeVideo(frames, currentBest)
                
            frames = []
            score = 0
        frames.append(f_t1)
        f_t1 = resizeAndDiscolor(f_t1)
        f_t1 = np.reshape(f_t1, [WIDTH, HEIGHT, 1])
        s_t1 = np.append(f_t1, s_t[:, :, :3], axis=2)
        s_t = s_t1
        t += 1
 
def main():
    sess = tf.InteractiveSession()
    frame, logits = model()
    infer(frame, logits, sess)
    
if __name__ == "__main__":
    main()
    
