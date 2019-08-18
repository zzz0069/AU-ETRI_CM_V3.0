import os

class AUETRIDataManager(object):

    def __init__(self):
        plot_train_val_loss
    
    # cache data
    def cache_data(data, path):
        if os.path.isdir(os.path.dirname(path)):
            file = open(path, 'wb')
            pickle.dump(data, file)
            file.close()
        else:
            print('Cache data doesnt exists.!')

    # restore data, use pickle.load to de-serialize a data stream
    def restore_data(path):
        data = dict()
        if os.path.isfile(path):
            file = open(path, 'rb')
            data = pickle.load(file)
        return data

    # save weights of model to h5 file for later use
    def save_model(model, path):
        json_string = model.to_json()
        if not os.path.isdir(path, 'cache'):
            os.mkdir(path, 'cache')
        open(os.path.join(path, 'cache', 'architecture.json'), 'w').write(json_string)
        model.save_weights(os.path.join(path, 'cache', 'model_weights.h5'), overwrite=True)

    # read previous saved model
    def read_model(path):
        model = model_from_json(open(os.path.join(path, 'cache', 'architecture.json')).read())
        model.load_weights(os.path.join(path, 'cache', 'model_weights.h5'))
        return model
