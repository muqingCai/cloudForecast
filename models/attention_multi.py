from utils.utilities import *


class Encoder(nn.Module):
    """
    encoder input: (seq_len, batch_size, input_size)
    decoder output:
    state:(h0, c0) (num_layers, batch_size, hidden_size)
    output: (seq_len, batch_size, hidden_size)
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout, is_cuda):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.is_cuda = is_cuda
        if is_cuda:
            self.lstm.cuda()

    def forward(self, input_data, state):
        output, state = self.lstm(input_data, state)
        # only output the last state
        return output, state

    def initState(self, batch_size):
        # initial the hidden state to zeros for all LSTMs
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0


class AttnDecoder(nn.Module):
    """
    decoder input:
    (1, batch_size, input_size)
    other input:
    decoder output: (batch_size, input_size)
    """

    def __init__(self, input_size, hidden_size, num_layers, time_steps, dropout, is_cuda):
        super(AttnDecoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.time_steps = time_steps
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers)
        self.in_lin = nn.Linear(input_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, time_steps)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.is_cuda = is_cuda
        self.out_lin = nn.Linear(hidden_size, input_size)
        if is_cuda:
            self.in_lin.cuda()
            self.lstm.cuda()
            self.out_lin.cuda()
            self.attn.cuda()
            self.attn_combine.cuda()

    def forward(self, input_data, state, encoder_outputs):
        input_data = self.in_lin(input_data)

        h, c = state[0], state[1]
        # input_data:(1, batch_size, input_size), hidden:(1, batch_size, hidden_size)=>(batch_size, input_size+hidden_size)
        com_hidden = torch.cat((input_data[0], h[0]), 1)
        attn_weights = F.softmax(self.attn(com_hidden), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), torch.transpose(encoder_outputs, 0, 1))
        com_input = torch.cat((input_data[0], torch.transpose(attn_applied, 0, 1)[0]), 1)

        com_input = self.attn_combine(com_input).unsqueeze(0)
        com_input = F.relu(com_input)

        output, state = self.lstm(com_input, state)
        # use the liner layer to transfer the dimension from hidden_size to input_size, get (batch_size,input_size)
        output = torch.sigmoid(self.out_lin(output.squeeze(0)))
        return output, state, attn_weights

    def initState(self, batch_size):
        # initial the hidden state to zeros for all LSTMs
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return h0, c0


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, steps_ahead, is_cuda):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.steps_ahead = steps_ahead
        self.is_cuda = is_cuda

    def forward(self, input_data, target_data, is_train):
        # get the input length
        input_len = input_data.size()[0]

        # get batch_size
        batch_size = input_data.size()[1]

        # get the input_size
        input_size = input_data.size()[2]

        encoder_state = self.encoder.initState(batch_size)

        # initialize the encoder outputs
        encoder_outputs = torch.zeros(input_len, batch_size, self.encoder.hidden_size)

        for i in range(input_len):
            # the reason of input_data[i].unsqueeze(0) is to get the LSTM input shape(seq_len, batch_size, input_size)
            encoder_output, encoder_state = self.encoder(input_data[i].unsqueeze(0), encoder_state)
            encoder_outputs[i] = encoder_output[0, 0]

        if self.is_cuda:
            encoder_outputs = encoder_outputs.cuda()

        # initial the decoder_input as zeros
        decoder_input = torch.zeros(1, batch_size, input_size)

        if self.is_cuda:
            decoder_input = decoder_input.cuda()

        # initialize the decoder outputs
        decoder_outputs = torch.zeros(self.steps_ahead, batch_size, input_size)

        if self.is_cuda:
            decoder_outputs = decoder_outputs.cuda()

        # get the last encoder state as decoder initial state
        H, C = encoder_state[0][-1].unsqueeze(0), encoder_state[1][-1].unsqueeze(0)
        decoder_state = (H, C)

        for i in range(self.steps_ahead):
            # get the output
            decoder_output, decoder_state, decoder_attention = self.decoder(decoder_input, decoder_state, encoder_outputs)

            # record the output
            decoder_outputs[i] = decoder_output

            # update the decoder_input
            if is_train:
                decoder_input = target_data[:, i].unsqueeze(0)
            else:
                decoder_input = decoder_output.unsqueeze(0)

        outputs = torch.transpose(decoder_outputs.squeeze(2), 0, 1)
        if self.is_cuda:
            outputs = outputs.cuda()
        return outputs


def evaluate(model, dataLoader, criterion, is_cuda):
    model.eval()
    pred_val = torch.tensor([])
    target_val = torch.tensor([])
    for j in range(len(dataLoader)):
        sample = dataLoader[j]
        sample_x = sample["x"]
        if len(sample_x) != 0:
            sample_x = np.stack(sample_x)
            inputs = Variable(torch.FloatTensor(sample_x), requires_grad=False)
            input_variable = torch.transpose(inputs, 0, 1)
            target_variable = Variable(torch.FloatTensor(sample["y"]), requires_grad=False)
            # use GPU
            if is_cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
            output = model(input_variable, None, False)
            pred_val = torch.cat((pred_val, output.cpu()), 0)
            target_val = torch.cat((target_val, target_variable.cpu()), 0)
    loss_val = criterion(pred_val, target_val)
    pred_val = np.array(pred_val.data.tolist())
    target_val = np.array(target_val.data.tolist())
    return loss_val, pred_val, target_val


def fit(model, dataLoader, n_epoch, criterion, optimizer, evaluate_point, is_cuda, directory):
    # global_loss_val is the infinity positive integer
    global_loss_val = np.inf

    # record the loss
    trainLoss, valLoss = list(), list()

    # get data
    trainLoader, valLoader = dataLoader[0], dataLoader[1]

    # begin training
    for i in range(n_epoch):

        # transfer the model to train()
        model.train()

        # record the result series
        pred_train = torch.tensor([])
        target_train = torch.tensor([])

        # iterate to train and train batch_size one iterator
        for j in range(len(trainLoader)):

            sample = trainLoader[j]
            sample_x = sample["x"]

            if len(sample_x) != 0:
                # stacked X, change the batch_size tuple(seq_len,input_size) to （batch_size,seq_len,input_size）
                sample_x = np.stack(sample_x)
                input_data = Variable(torch.FloatTensor(sample_x), requires_grad=False)

                # transpose，swap the first the second dimension，because the inputs of LSTM is (seq_len, batch_size, input_size),
                # which means the length of inputs, batch_size and features size, respectively
                input_variable = torch.transpose(input_data, 0, 1)
                target_variable = Variable(torch.FloatTensor([x for x in sample["y"]]), requires_grad=False)

                # use GPU
                if is_cuda:
                    input_variable = input_variable.cuda()
                    target_variable = target_variable.cuda()

                optimizer.zero_grad()

                output = model(input_variable, target_variable, True)

                loss = criterion(output, target_variable)

                # loss back propagation
                loss.backward()

                # update the network
                optimizer.step()

                pred_train = torch.cat((pred_train, output.cpu()), 0)
                target_train = torch.cat((target_train, target_variable.cpu()), 0)

        # calculate the whole loss with mean_squared_loss criterion
        loss_train = criterion(pred_train, target_train)

        # todo delete
        # if i % 100 == 0 and i >= 100:
        #     pred_train = pred_train.view(-1, pred_train.size()[2])
        #     target_train = target_train.view(-1, target_train.size()[2])
        #     pred_val = np.array(pred_train.tolist())[:, 0]
        #     target_val = np.array(target_train.tolist())[:, 0]
        #     plt.figure(figsize=(20, 8))
        #     plt.plot(pred_val[:360], color='red')
        #     plt.plot(target_val[:360], color='blue')
        #     plt.show()

        print("Epoch %d/%d | train loss: %.6f " % (i, n_epoch, loss_train))

        # evaluate and save model
        if i % evaluate_point == 0 and i >= 50:

            print("\ni: " + str(i))
            loss_val, pred_val, target_val = evaluate(model, valLoader, criterion, is_cuda)

            # save according to loss_val
            if loss_val < global_loss_val:
                print("CURRENT BEST")
                global_loss_val = loss_val
                save_checkpoint({'epoch': i, 'state_dict': model.state_dict()}, directory, 'Seq2Seq', True)

            print("LOSS TRAIN: " + str(float(loss_train)))
            print("LOSS VAL: " + str(float(loss_val)))
            trainLoss.append(float(loss_train))
            valLoss.append(float(loss_val))

        if i == n_epoch - 1:
            save_checkpoint({'epoch': i, 'state_dict': model.state_dict()}, directory, 'Seq2Seq', False)

    return trainLoss, valLoss
