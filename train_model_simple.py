import json
import sys
import os
import os.path as pth
import vgg
import tensorflow as tf
import numpy as np
import time
import layers as L
import tools
import dataset
import yaml
import argparse


# =====================================
# Training configuration default params
# =====================================
config = {}

#################################################################

# customize your model here
# =========================
def build_model(input_data_tensor, input_label_tensor, train_mode):
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]

    # images = tf.image.resize_images(input_data_tensor, [224, 224], method=0, align_corners=False)
    images = input_data_tensor

    logits = vgg.build(images, n_classes=num_classes, training=train_mode)
    probs = tf.nn.softmax(logits)
    loss_classify = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    loss_weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))
    loss = loss_classify + weight_decay * loss_weight_decay
    error_top5 = L.topK_error(probs, input_label_tensor, K=5)
    error_top1 = L.topK_error(probs, input_label_tensor, K=1)

    # you must return a dictionary with loss as a key, other variables
    return dict(loss=loss,
                probs=probs,
                logits=logits,
                error_top5=error_top5,
                error_top1=error_top1)


def train(trn_data, tst_data=None):
    learning_rate = config['learning_rate']
    experiment_dir = config['experiment_dir']
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config["num_samples_per_epoch"]
    steps_per_epoch = num_samples_per_epoch // batch_size
    num_steps = steps_per_epoch * num_epochs
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    train_log_fpath = pth.join(experiment_dir, 'train.log')
    vld_iter = config["vld_iter"]
    checkpoint_iter = config["checkpoint_iter"]
    worker_number = config["worker_number"]
    pretrained_weights = config.get("pretrained_weights", None)

    # ========================
    # construct training graph
    # ========================
    G = tf.Graph()
    with G.as_default():
        input_data_tensor = tf.placeholder(tf.float32, [None] + data_dims, name='input_data_tensor')
        input_label_tensor = tf.placeholder(tf.int32, [None], name='input_label_tensor')
        learning_rate = tf.placeholder(tf.float32, name='lr_placeholder')
        train_mode = tf.placeholder(tf.bool, name='train_mode_placeholder')
        model = build_model(input_data_tensor, input_label_tensor, train_mode)
        var_list = tf.trainable_variables()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # grads_and_vars = optimizer.compute_gradients(model["loss"])
        grads = tf.gradients(model["loss"], var_list)
        clipped_grads = [tf.clip_by_norm(grad, 1) for grad in grads]
        aggregated_grads = [tf.placeholder(tf.float32, var.get_shape()) \
                                    for var in var_list]
        grad_step = optimizer.apply_gradients(zip(aggregated_grads, var_list))
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()


    # ===================================
    # initialize and run training session
    # ===================================
    log = tools.StatLogger(train_log_fpath)
    # config_proto = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(graph=G, config=config_proto)
    sess = tf.Session(graph=G)
    sess.run(init)
    # tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        if pretrained_weights:
            print("-- loading weights from %s" % pretrained_weights)
            tools.load_weights(G, pretrained_weights)

        train_x = trn_data[0]
        train_y = trn_data[1]
        idx = np.arange(num_samples_per_epoch, dtype=np.int32)

        for epoch in range(1, num_epochs+1):
            lr = 0.1 / float(1 << (epoch / 25))
            np.random.shuffle(idx)
            total_loss, total_acc, total_acc_5 = 0.0, 0.0, 0.0
            print 'epoch %d' % epoch
            for step in range(steps_per_epoch):
                X_trn = train_x[idx[step * batch_size: (step + 1) * batch_size]]
                Y_trn = train_y[idx[step * batch_size: (step + 1) * batch_size]]

                step_loss, step_acc, step_acc_5 = 0.0, 0.0, 0.0
                grad_workers = []
                for worker in range(worker_number):
                    micro_batch_size = batch_size / worker_number
                    micro_x_trn = X_trn[worker * micro_batch_size: (worker + 1) * micro_batch_size]
                    micro_y_trn = Y_trn[worker * micro_batch_size: (worker + 1) * micro_batch_size]

                    ops = [clipped_grads] + [model[k] for k in sorted(model.keys())]
                    inputs = {input_data_tensor: micro_x_trn, input_label_tensor: micro_y_trn,
                             learning_rate: lr, train_mode: True}
                    results = sess.run(ops, feed_dict=inputs)
                    grad_workers.append([grad for grad in results[0]])
                    results = dict(zip(sorted(model.keys()), results[1:]))
                    step_loss += results["loss"]
                    step_acc += 1 - results["error_top1"]
                    step_acc_5 += 1 - results["error_top5"]

                #################################################
                # Aggregate Gradients from different workers
                #################################################
                # grad_workers: [[(1,'t'), (2, 'tt'), (3, 'ttt')], [(4,'t'), (5, 'tt'), (6, 'ttt')],
                #                [(7,'t'), (8, 'tt'), (9, 'ttt')]]
                # zip(*grad_workers):[((1, 't'), (4, 't'), (7, 't')), ((2, 'tt'), (5, 'tt'), (8, 'tt')),
                #                ((3, 'ttt'), (6, 'ttt'), (9, 'ttt'))]
                grad_workers = [pair for pair in zip(*grad_workers)]
                new_grad = []
                for grad in grad_workers:
                    variance = np.var(np.asarray(grad), axis=0)
                    avg_variance = np.mean(variance)
                    scale = np.divide(variance, avg_variance)
                    final_scale = np.exp(1 - scale)

                    avg_grad = np.mean(np.asarray(grad), axis=0)
                    final_grad = np.multiply(final_scale, avg_grad)
                    new_grad.append(final_grad)

                # new_grad = [np.mean(np.asarray(grad), axis=0) for grad in grad_workers]

                #################################################
                # Run Gradients Updates
                #################################################
                grad_dict = {}
                grad_dict[learning_rate] = lr
                for i in range(len(aggregated_grads)):
                    grad_dict[aggregated_grads[i]] = new_grad[i]
                sess.run(grad_step, feed_dict=grad_dict)

                avg_step_loss = step_loss / worker_number
                avg_step_acc = step_acc / worker_number
                avg_step_acc_5 = step_acc_5 / worker_number

                total_loss += avg_step_loss
                total_acc += avg_step_acc
                total_acc_5 += avg_step_acc_5

                ####################################
                # Old step one worker workflow
                ####################################
                # ops = [grad_step] + [model[k] for k in sorted(model.keys())]
                # inputs = {input_data_tensor: X_trn, input_label_tensor: Y_trn,
                #          learning_rate: lr, train_mode: True}
                # results = sess.run(ops, feed_dict=inputs)
                # results = dict(zip(sorted(model.keys()), results[1:]))
                # total_loss += results["loss"]
                # total_acc += 1 - results["error_top1"]

                tools.update_progress(step * 1.0 / steps_per_epoch,
                                    'training loss = %f, accuracy = %f' % (avg_step_loss,
                                                            avg_step_acc))

                log.report(epoch=epoch,
                           step=step,
                           split="TRN",
                        #    probs=str(results["probs"]),
                        #    labels=str(Y_trn),
                           error_top1=float(avg_step_acc),
                           error_top5=float(avg_step_acc_5),
                           loss=float(avg_step_loss))

            info = '\ntraining loss = %f, training accuracy = %f, lr = %f' \
                % (total_loss / steps_per_epoch, total_acc / steps_per_epoch, lr)
            print info

            print("-- running test on test split")
            X_tst = tst_data[0]
            Y_tst = tst_data[1]
            inputs = [input_data_tensor, input_label_tensor, train_mode]
            args = [X_tst, Y_tst, False]
            ops = [model[k] for k in sorted(model.keys())]
            results = tools.iterative_reduce(ops, inputs, args, batch_size=batch_size, fn=lambda x: np.mean(x, axis=0))
            results = dict(zip(sorted(model.keys()), results))
            print("Test Epoch:%-5d, error_top1: %.4f, error_top5: %.4f, loss:%s" % (epoch,
                                                                                 results["error_top1"],
                                                                                 results["error_top5"],
                                                                                 results["loss"]))
            log.report(epoch=epoch,
                       split="TST",
                       error_top1=float(results["error_top1"]),
                       error_top5=float(results["error_top5"]),
                       loss=float(results["loss"]))

        if (epoch % checkpoint_iter == 0):
            print("-- saving check point")
            tools.save_weights(G, pth.join(checkpoint_dir, "weights.%s" % epoch))



        # Start training loop
        # for step in range(1, num_steps):
        #     batch_train = trn_data_generator.next()
        #     X_trn = np.array(batch_train[0])
        #     Y_trn = np.array(batch_train[1])
        #
        #     ops = [grad_step] + [model[k] for k in sorted(model.keys())]
        #     inputs = {input_data_tensor: X_trn, input_label_tensor: Y_trn}
        #     results = sess.run(ops, feed_dict=inputs)
        #     results = dict(zip(sorted(model.keys()), results[1:]))
        #     print("TRN step:%-5d, error_top5: %s, error_top1: %s, loss:%s" % (step,
        #                                                             results["error_top5"],
        #                                                             results["error_top1"],
        #                                                             results["loss"]))
        #     log.report(step=step,
        #                split="TRN",
        #                probs=str(results["probs"]),
        #                labels=str(Y_trn),
        #                error_top5=float(results["error_top5"]),
        #                error_top1=float(results["error_top1"]),
        #                loss=float(results["loss"]))
        #
        #     # report evaluation metrics every 10 training steps
        #     if (step % vld_iter == 0):
        #         print("-- running evaluation on vld split")
        #         X_vld = vld_data[0]
        #         Y_vld = vld_data[1]
        #         inputs = [input_data_tensor, input_label_tensor]
        #         args = [X_vld, Y_vld]
        #         ops = [model[k] for k in sorted(model.keys())]
        #         results = tools.iterative_reduce(ops, inputs, args, batch_size=1, fn=lambda x: np.mean(x, axis=0))
        #         results = dict(zip(sorted(model.keys()), results))
        #         print("VLD step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (step,
        #                                                                              results["error_top1"],
        #                                                                              results["error_top5"],
        #                                                                              results["loss"]))
        #         log.report(step=step,
        #                    split="VLD",
        #                    error_top5=float(results["error_top5"]),
        #                    error_top1=float(results["error_top1"]),
        #                    loss=float(results["loss"]))

            # if (step % checkpoint_iter == 0) or (step + 1 == num_steps):
            #     print("-- saving check point")
            #     tools.save_weights(G, pth.join(checkpoint_dir, "weights.%s" % step))

def main():
    batch_size = config['batch_size']
    experiment_dir = config['experiment_dir']

    # setup experiment and checkpoint directories
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # trn_data_generator, vld_data = dataset.get_cifar10(batch_size)
    trn_data, tst_data = dataset.get_cifar10(batch_size)
    train(trn_data, tst_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

        print "Experiment config"
        print "------------------"
        print json.dumps(config, indent=4)
        print "------------------"
    main()
