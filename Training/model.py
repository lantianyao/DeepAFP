from tensorflow.keras.layers import Dense, Input, Flatten, Dropout,Bidirectional, LSTM, GRU,Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def model_multiview_aafeature_bert(filter_num=64, kernel_size=3, lstm_num=30):
    bert = Input(shape=(768))
    aa_feature = Input(shape=(100, 45))
    bert_layer = Dense(300, activation='relu')(bert)

    cnn = Conv1D(filters=filter_num, kernel_size=kernel_size, padding='same', strides=1, activation='relu')(aa_feature)
    cnn = MaxPooling1D(pool_size=2, strides=2)(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = Bidirectional(LSTM(lstm_num, return_sequences=True))(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(100, activation='relu')(cnn)

    cnn1 = Conv1D(filters=filter_num, kernel_size=kernel_size + 1, padding='same', strides=1, activation='relu')(aa_feature)
    cnn1 = MaxPooling1D(pool_size=2, strides=2)(cnn1)
    cnn1 = Dropout(0.5)(cnn1)
    cnn1 = Bidirectional(LSTM(lstm_num, return_sequences=True))(cnn1)
    cnn1 = Flatten()(cnn1)
    cnn1 = Dense(100, activation='relu')(cnn1)

    cnn2 = Conv1D(filters=filter_num, kernel_size=kernel_size + 2, padding='same', strides=1, activation='relu')(aa_feature)
    cnn2 = MaxPooling1D(pool_size=2, strides=2)(cnn2)
    cnn2 = Dropout(0.5)(cnn2)
    cnn2 = Bidirectional(LSTM(lstm_num, return_sequences=True))(cnn2)
    cnn2 = Flatten()(cnn2)
    cnn2 = Dense(100, activation='relu')(cnn2)

    merge = concatenate([bert_layer, cnn, cnn1, cnn2], axis=1)
    x = Dense(64, activation='relu')(merge)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=[bert, aa_feature], outputs=output)
    adam = Adam(learning_rate=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
