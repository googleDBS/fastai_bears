from fastai.vision import *

def predict_image():
    # Location of bucket with image
    path = Path('tutorials/fastai')
    url = path/'images'/'bear.jpg'

    # Download Image
    img = open_image(url)

    # Load model from
    path_to_model = path/'models'
    learn = load_learner(path_to_model)

    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class

print(predict_image())