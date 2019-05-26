import time
import data_util
import score_model
import loss
from absl import flags
import sys
import os
import logging
import glob
import numpy as np

          
# TODO: to repalce loss by listwise loss
          
strtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
print("Exp code: ", strtime)
workdir = os.path.split(os.path.realpath(__file__))[0]
             
# create experiment dir
experiment_dir = os.path.join(workdir, "experiments/%s"%strtime)
# create model dir in this experiment
model_dir = os.path.join(experiment_dir, "models")
    

# make experiment dir for save everything for this experiemnt
os.mkdir(experiment_dir)
os.mkdir(model_dir) 
    
#logfile = os.path.join(experiment_dir, "training_log.log")

#handlers = [logging.FileHandler(os.path.join(experiment_dir, "training_log.log")), logging.StreamHandler(sys.stdout)]

#logging.getLogger('keras').handlers = handlers

flags.DEFINE_string("model_name", None, "required, a key in score_model.py --> model_dict")

flags.DEFINE_string("train_dataset", os.path.join(workdir, "voice_v10_train"), "Input file path used for training.")
flags.DEFINE_string("vali_dataset", os.path.join(workdir, "voice_v10_val"), "Input file path used for validation.")
flags.DEFINE_string("test_dataset", os.path.join(workdir, "../../dataset/kg_tmb_score_for_predict"), "Input file path used for testing.")
flags.DEFINE_string("output_dir", model_dir, "Output directory for models.")
flags.DEFINE_string("devices", "2", "devices used, splited by quotation")
flags.DEFINE_string("note","","anything you want to say about this experiment, do not use space")


flags.DEFINE_integer("batch_size", 32, "The batch size for training.")

flags.DEFINE_integer("feature_timestep", 5167, "Number of features per document. Set 0 for vairous timestep")
flags.DEFINE_integer("feature_dim", 130, "dim for each timestep in feature.")
flags.DEFINE_integer("list_size", 64, "List size used for training.")
flags.DEFINE_integer("epochs", 100, "number of training epochs")
flags.DEFINE_integer("n_bin", 5, "number of bins, 2 for only tail/head data.")

flags.DEFINE_enum("makelist_method", "random", ["bin","random","quantbin", "normbin"], "random or quantbin")
flags.DEFINE_string("opt", "RMSprop", "optimizer used")
flags.DEFINE_string("loss", "ranknet",
                    "The RankingLossKey for loss function.")
flags.DEFINE_integer("ltr_topk", 0, "topk for ltr model")
flags.DEFINE_boolean("act", False, "Add sigmoid activation at the end of the model?")
flags.DEFINE_float("lr", 0.002, "learning rate")
flags.DEFINE_boolean("test_overfitting", False, "Test overfitting, if True, will use training set to evaluate")
flags.DEFINE_string("feature_func","melmix", "feature func defined in feature_util.py")
flags.DEFINE_string("clipgrad","norm", "clip gradient")
flags.DEFINE_float("lr_decay", 0.2, "decay lr every 5 epoch")

flags.mark_flag_as_required("model_name")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] =FLAGS.devices



import dataset
import feature_util


feature_func = feature_util.get_feature_extract_func(FLAGS.feature_func)

def train():
    #input_shape = (FLAGS.list_size, FLAGS.feature_timestep, FLAGS.feature_dim)

    input_shape = (FLAGS.feature_timestep, FLAGS.feature_dim)

    # collect flags
    flag_string = FLAGS.flags_into_string()
    # dump flag_string into experiment_dir
    with open(os.path.join(experiment_dir, "config.conf"), "w") as f:
        f.write(flag_string)

    #TODO
    from keras.models import model_from_json
    model_json = "".join(open('/data/home/v_yunyigao/moyannet/experiments/20190524170741/models/model_architecture').readlines())
    model = model_from_json(model_json)
    #model = score_model.get_model(FLAGS.model_name, input_shape, act=FLAGS.act)

    model.load_weights("/data/home/v_yunyigao/moyannet/experiments/20190524170741/models/model_25.h5")

    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #test_gen = data_util.test_datagen(test_flist, target_ts=FLAGS.feature_timestep,
    #                                feature_dim=FLAGS.feature_dim)
    #TODO
    with open('/data/home/v_yunyigao/moyannet/results/vocal_1__kg_t50_t100_20190524170741__model_25_predcit.xls','w') as f:
        for fpath in np.sort(glob.glob('/data2/siyuzhou/VOICE/dataset/kg_t50_t100_for_predict/*')):
            try:
                feat = feature_func(fpath)
                feat = data_util.clip_feat(feat, None, FLAGS.feature_dim)
                #feat = data_util.clip_feat(feat, FLAGS.feature_timestep, FLAGS.feature_dim)
                feat = np.expand_dims(feat,axis=0)
                preds = model.predict(feat)[0]

                #out = int(preds[1]*100)
                out = int(preds*100)
                pscore = fpath.split('_')[-1][:-4]
                f.write('%s\t%s\t%d\n'%(fpath,pscore,out))
                #print (np.argmax(preds),preds, out)
                print (preds, out)
            except:
                pass

    # prepare callbacks
    from callbacks import LTRcallback
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from data_util import val_datagen
    import functools

    val_gen = functools.partial(val_datagen, target_ts=FLAGS.feature_timestep, feature_dim=FLAGS.feature_dim)

    print(val_gen)

    model_path = os.path.join(model_dir, "model_{epoch:02d}.h5")
    mckp = ModelCheckpoint(model_path, period=1,monitor='val_acc', save_weights_only=True)

    def lr_decay(e,lr):
        if e%5==0:
            lr = lr*FLAGS.lr_decay
        return lr

    lrsd = LearningRateScheduler(lr_decay)
    if FLAGS.test_overfitting:
        overfitting_test_cbk = LTRcallback(val_gen, train_flist, experiment_dir)
        cbks = [overfitting_test_cbk, mckp, lrsd]
    else:
        cbks = [LTRcallback(val_gen, val_flist, experiment_dir), mckp, lrsd]

    model_graph = model.to_json()
    model_json_path = os.path.join(model_dir, "model_architecture")
    with open(model_json_path, "w") as g:
        g.write(model_graph)
    #steps_per_epoch = int(len(train_flist) / FLAGS.batch_size)
    #validation_steps = int(len(val_flist)/FLAGS.batch_size)
    #model.fit_generator(train_seq, epochs=FLAGS.epochs,callbacks=cbks, workers=32)#,use_multiprocessing=True, workers=10)
    model.fit_generator(train_seq, epochs=FLAGS.epochs, callbacks=[mckp],
                        validation_data=val_seq, workers=32)#,use_multiprocessing=True, workers=10)
    #model.fit_generator(train_seq, steps_per_epoch = 10, epochs=FLAGS.epochs,workers=16,callbacks=cbks)




if __name__ == "__main__":
    train()
