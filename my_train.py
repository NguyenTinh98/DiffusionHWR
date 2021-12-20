import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import my_utils
import my_nn
import time
import argparse
import my_preprocessing

def show_images(model, width, textstring='Ví dụ 123, mẫu 456', img_path=None):
    sourcename = 'assets/writer6.png'
    org_img = my_preprocessing.read_img(sourcename, 96)
    org_img = my_utils.pad_img(org_img, width, 96)
    writer_img = tf.expand_dims(org_img, 0)
    style_extractor = my_nn.StyleExtractor()

    style_vector = style_extractor(writer_img)
    tokenizer = my_utils.Tokenizer()
    beta_set = my_utils.get_beta_set()
    timesteps = len(textstring) * 16

    my_utils.run_batch_inference(model, beta_set, textstring, style_vector, 
                            tokenizer=tokenizer, time_steps=timesteps, diffusion_mode='new', 
                            show_samples=True, path=img_path)

@tf.function
def train_step(x, pen_lifts, text, style_vectors, glob_args):
    model, alpha_set, bce, train_loss, optimizer, score_loss, pl_loss, penalty = glob_args
    alphas = my_utils.get_alphas(len(x), alpha_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alphas) * x 
    x_perturbed += tf.sqrt(1 - alphas) * eps
    
    with tf.GradientTape() as tape:
        score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
        loss, l1, l2 = my_nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce, penalty)
    
    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    score_loss(l1)
    pl_loss(l2)
    return score, att

def train(dataset, iterations, model, optimizer, alpha_set, ff, MODEL_PATH, WIDTH, penalty = 1, print_every=1000, save_every=10000):
    s = time.time()
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    train_loss = tf.keras.metrics.Mean()
    score_loss = tf.keras.metrics.Mean()
    pl_loss = tf.keras.metrics.Mean()

    min_loss = 1000
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        glob_args = model, alpha_set, bce, train_loss, optimizer, score_loss, pl_loss, penalty
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors, glob_args)
      
        if optimizer.iterations%print_every==0:
            temp = ("Iteration %d, Loss %f, Score %f, Pl %f, Time %ds" % (optimizer.iterations, train_loss.result(), score_loss.result(), pl_loss.result(), time.time()-s))
            print(temp)
            show_images(model, WIDTH)
            with open(ff, 'a') as f:
              f.write(temp + '\n')
            train_loss.reset_states()
            score_loss.reset_states()
            pl_loss.reset_states()

        if (optimizer.iterations+1) % save_every==0:
          if min_loss > train_loss.result():
            # save_path = ckpt_path + './weights/model_step%d.h5' % (optimizer.iterations+1)
            save_path = MODEL_PATH[:-3] + '_step%d' % (optimizer.iterations+1) + '.h5'
            model.save_weights(save_path)
            min_loss = train_loss.result()
            
        if optimizer.iterations > iterations:
            model.save_weights(MODEL_PATH)
            break


def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--steps', help='number of trainsteps, default 60k', default=20000, type=int)
    parser.add_argument('--batchsize', help='default 96', default=32, type=int)
    parser.add_argument('--seqlen', help='sequence length during training, default 480', default=3000, type=int)
    parser.add_argument('--textlen', help='text length during training, default 50', default=100, type=int)
    parser.add_argument('--width', help='offline image width, default 1400', default=2100, type=int)
    parser.add_argument('--warmup', help='number of warmup steps, default 10k', default=4000, type=int)
    parser.add_argument('--dropout', help='dropout rate, default 0', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='number of channels in first layer, default 128', default=128, type=int)
    parser.add_argument('--print_every', help='show train loss every n iters', default=500, type=int)
    parser.add_argument('--save_every', help='save ckpt every n iters', default=4000, type=int)
    parser.add_argument('--log_train', help='save loss every n iters', default='log_train.txt', type=str)
    parser.add_argument('--path_dts', help='path to dataset', default='data/my_train_strokes.p', type=str)
    parser.add_argument('--path_model', help='path to save model', default='./weights/my_model.h5', type=str)
    parser.add_argument('--limit', help='limit for strokes points', default=15, type=int)
    parser.add_argument('--penalty', help='penalty for score_loss', default=1.0, type=float)


    args = parser.parse_args()
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batchsize
    MAX_SEQ_LEN = args.seqlen
    MAX_TEXT_LEN = args.textlen
    WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    SAVE_EVERY = args.save_every
    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    MAX_SEQ_LEN = MAX_SEQ_LEN - (MAX_SEQ_LEN%8) + 8
    MODEL_PATH = args.path_model

    BUFFER_SIZE = 3000
    L = 60
    tokenizer = my_utils.Tokenizer()
    beta_set = my_utils.get_beta_set()
    alpha_set = tf.math.cumprod(1-beta_set)

    style_extractor = my_nn.StyleExtractor(96, WIDTH, 3)
    model = my_nn.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, d_emb= len(tokenizer.chars), drop_rate=DROP_RATE)
    lr = my_nn.InvSqrtSchedule(C3, warmup_steps=WARMUP_STEPS)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, clipnorm=100)
    

    path = args.path_dts
    limit = args.limit
    strokes, texts, samples = my_utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, 96, limit)
    print(samples.shape)
    dataset = my_utils.create_dataset(strokes, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE)
    print(samples.shape)
    print('----training---')
    train(dataset, NUM_STEPS, model, optimizer, alpha_set, args.log_train, MODEL_PATH, WIDTH, args.penalty, PRINT_EVERY, SAVE_EVERY)

if __name__ == '__main__':
    main()
