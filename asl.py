from models.attention_multi import *
from datetime import datetime
from utils.utilities import *
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def parse_args():
    # the parameter of predictor
    parser = argparse.ArgumentParser(description="Run ASL.")
    parser.add_argument('--dataset', type=str, default='sortM1_5000',
                        help='the dataset of time series.')
    parser.add_argument('--time_steps', type=int, default=12,
                        help='the input length of time series.')
    parser.add_argument('--steps_ahead', type=int, default=3,
                        help='the predict length of time series.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='the batch_size for training.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='the learning rate.')
    parser.add_argument('--epochs', type=int, default=1001,
                        help='the epochs for training.')
    parser.add_argument('--encoder_hidden_size', type=int, default=32,
                        help='the encoder hidden unit.')
    parser.add_argument('--encoder_stack_layers', type=int, default=3,
                        help='the layer number stacked lstm.')
    parser.add_argument('--decoder_stack_layers', type=int, default=1,
                        help='the layer number stacked lstm.')
    parser.add_argument('--evaluate_point', type=int, default=50,
                        help='the lstm evaluate_point for evaluating.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the dropout for lstm training.')
    parser.add_argument('--is_shuffle', type=bool, default=True,
                        help='whether to shuffle data.')
    parser.add_argument('--valSize', type=int, default=720,
                        help='the rate of valuation data.')
    parser.add_argument('--testSize', type=int, default=720,
                        help='the rate of validation data.')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='the running mode, train or test.')
    parser.add_argument('--is_cuda', type=bool, default=True,
                        help='use cuda or not.')

    return parser.parse_args()


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------
    # --------------------------- STEP 0: load the configuration of parameter----------------
    # ---------------------------------------------------------------------------------------
    args = parse_args()
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    encoder_hidden_size = args.encoder_hidden_size
    decoder_hidden_size = encoder_hidden_size
    encoder_stack_layers = args.encoder_stack_layers
    decoder_stack_layers = args.decoder_stack_layers
    evaluate_point = args.evaluate_point
    dropout = args.dropout
    is_shuffle = args.is_shuffle
    dataset = args.dataset
    time_steps = args.time_steps
    steps_ahead = args.steps_ahead
    steps_move = steps_ahead
    valSize = args.valSize
    testSize = args.testSize
    is_train = args.is_train
    is_cuda = args.is_cuda
    start_time = datetime.now()
    strTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if is_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # if train, create new model, else load the model
    if is_train:
        fileName = '%s/%s_step%s_ahead%s' % (dataset, strTime, str(time_steps), str(steps_ahead))
    else:
        testModel = '2020-05-15-21-07-54_step12_ahead3'
        fileName = '%s/%s' % (dataset, testModel)

    directoryModel = "runs/cloud/%s/model/" % fileName
    directoryRESULT = "runs/cloud/%s/result/" % fileName

    # ---------------------------------------------------------------------------
    # --------------------------- LOAD DATA -------------------------------------
    # ---------------------------------------------------------------------------

    # dataset
    path = "data/cloudData/machineData/%s.csv" % dataset

    # read data
    feats = pd.read_csv(path)

    # get the useful columns
    feats = feats.iloc[:, 3:]

    # the dimension of features
    input_size = len(feats.columns)

    # get the data of training set, validate set and test set
    val_data = feats[:valSize]
    train_data = feats[valSize:-testSize]
    test_data = feats[-testSize:]

    # define different scaler for train, validate, test
    train_scaler = MinMaxScaler(feature_range=(0, 1))
    val_scaler = MinMaxScaler(feature_range=(0, 1))
    test_scaler = MinMaxScaler(feature_range=(0, 1))

    # normalize the train set, validate set and test set
    norm_train = normalLizeData(train_data, False, train_scaler)
    norm_val = normalLizeData(val_data, False, train_scaler)
    norm_test = normalLizeData(test_data, False, test_scaler)

    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    # mean squared error loss
    criterion = nn.MSELoss()

    # Instantiate the model
    encoder = Encoder(input_size, encoder_hidden_size, encoder_stack_layers, dropout, is_cuda)
    decoder = AttnDecoder(input_size, decoder_hidden_size, decoder_stack_layers, time_steps, dropout, is_cuda)
    model = Seq2Seq(encoder, decoder, steps_ahead, is_cuda)

    # record the true values and predict values
    y_test_list = test_data.values
    y_preds_list = list()

    # validate the features of inverse
    # inverse_test = inverseData(norm_y_test, True, y_test_scaler)

    # ---------------------------------------------------------------------------
    # --------------------------- TRAIN -----------------------------------------
    # ---------------------------------------------------------------------------
    if is_train:
        # get the train_extra, val_extra and test_extra for generating LSTM inputs
        val_extra = np.concatenate((norm_train[-time_steps:], norm_val))

        # divide the dataset to get the LSTM input unit and output unit
        x_train_lstm, y_train_lstm, = prepare_data_lstm(norm_train, norm_train, time_steps, steps_ahead,
                                                        steps_move, True)
        x_val_lstm, y_val_lstm = prepare_data_lstm(val_extra, norm_val, time_steps, steps_ahead, steps_move,
                                                   False)

        # ---------------------------------------------------------------------------
        # ------------- STEP 6: TIME-SERIES REGRESSION USING seq2seq model(encoder:
        # --------------stacked LSTMs, decoder: single LSTM and single liner layer
        # ---------------------------------------------------------------------------

        # shuffle the dataset
        combined = list(zip(x_train_lstm, y_train_lstm))
        if is_shuffle:
            random.shuffle(combined)
        x_train_lstm, y_train_lstm = zip(*combined)

        # get the dataLoader
        trainLoader = ExampleDataset(x_train_lstm, y_train_lstm, batch_size)
        valLoader = ExampleDataset(x_val_lstm, y_val_lstm, batch_size)
        dataLoader = (trainLoader, valLoader)

        # optimize for updating the weight of the network
        optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)
        fit(model, dataLoader, epochs, criterion, optimizer, evaluate_point, is_cuda, directoryModel)

    # ---------------------------------------------------------------------------
    # --------------------------- TEST ------------------------------------------
    # ---------------------------------------------------------------------------
    else:
        resumeSeq2Seq = directoryModel + 'last.pth.tar'
        if os.path.isfile(resumeSeq2Seq):
            print("=> loading checkpoint Seq2SeqModel '{}'".format(resumeSeq2Seq))
            Seq2SeqModel_checkpoint = torch.load(resumeSeq2Seq)
            model.load_state_dict(Seq2SeqModel_checkpoint['state_dict'])
            print("=> loaded checkpoint (Seq2SeqModel '{}' with epoch {})"
                  .format(resumeSeq2Seq, Seq2SeqModel_checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resumeSeq2Seq))

        # transfer the mode to eval
        model.eval()

        # begin iterator to get predict result
        num_iterations = int(testSize / steps_ahead)
        test_extra = np.concatenate((norm_val[-time_steps:], norm_test))
        for n in range(num_iterations):
            print("the current iterator is %s" % n)
            test_x = test_extra[n * steps_ahead:n * steps_ahead + (time_steps + steps_ahead)]
            test_y = test_x[-steps_ahead:]

            x_test_lstm, y_test_lstm, = prepare_data_lstm(test_x, test_y, time_steps, steps_ahead,
                                                          steps_move, False)
            testLoader = ExampleDataset(x_test_lstm, y_test_lstm, batch_size)
            loss_test, pred_test, target_test = evaluate(model, testLoader, criterion, is_cuda)
            print("LOSS TEST: " + str(float(loss_test)) + "\n")

            y_preds_list.extend(pred_test)

        # change the shape
        y_preds_list = np.array(y_preds_list)
        y_preds_list = y_preds_list.reshape(y_preds_list.shape[0] * y_preds_list.shape[1], -1)
        # inverse data
        y_preds_list = inverseData(y_preds_list, False, test_scaler)
        # nameList
        nameList = ['cpu_util_precent', 'mem_util_percent', 'net_in', 'net_out', 'disk_io_percent']
        MultiSaveResult(y_test_list, y_preds_list, nameList, directoryRESULT)
        # save the configuration
        args2Dict = vars(args)
        save_parameters(directoryRESULT, args2Dict)
