

def train_model(x,
                y,
                model,
                name="noname",
                tboard=False,
                ckpt=False,
                epochs=10,
                batch=3,
                val_split=0.3,
                estop=False,
                estop_patience=10,
                estop_min_delta=0.0001,
                estop_monitor='val_acc'
                ):

    from datetime import datetime
    import os
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import EarlyStopping

    callbacks = []


    if estop is True:
        earlystop_callback = EarlyStopping(
          monitor=estop_monitor, min_delta=estop_min_delta,
          patience=estop_patience)
        callbacks.append(earlystop_callback)
        pass


    if tboard is True:
        logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S-") + name
        if not os.path.exists('logs'):
            os.makedirs('logs')

        os.mkdir(logdir)
        logdir = os.path.join(logdir)

        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=1,
                                           profile_batch=100000000)

        callbacks.append(tensorboard_callback)
        pass

    if ckpt is True:

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        ckpf_dir = os.path.join('checkpoints',
                            datetime.now().strftime("%Y%m%d-%H%M%S-") + name,
                            )
        os.makedirs(ckpf_dir)

        ckpf = os.path.join('checkpoints',
                            datetime.now().strftime("%Y%m%d-%H%M%S-") + name,
                            name + '.hdf5')

        checkpointer = ModelCheckpoint(filepath=ckpf,
                                       verbose=1,
                                       save_best_only=True)

        callbacks.append(checkpointer)
        pass

    hist = model.fit(x,
                     y,
                     batch_size=batch,
                     epochs=epochs,
                     validation_split=val_split,
                     # validation_data=(x_test, y_test_one_hot),
                     callbacks=callbacks,
                     )
    return hist
