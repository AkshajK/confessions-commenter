import requests
import urllib 
import shutil
import os 
import random

def download_model(model_path="./meme_generator_model.h5", params_path="./params_generator_model.json", meme_data_path="./meme_data.json"):
    if os.path.exists(model_path) and os.path.exists(params_path) and os.path.exists(meme_data_path):
        # print("You already have downloaded the model!")
        return 
    import gdown
    print("First time! Downloading model. This may take a minute. (will only happen once)")
    gdown.download("https://drive.google.com/uc?id=19AggyW1LK5ME3NaW4wkygpuki-Cqp8s-", model_path, quiet=False)
    gdown.download("https://drive.google.com/uc?id=1Sm_-HBTNGOdSx6QntbAo-_jzGNCxKIdP", params_path , quiet=False)
    gdown.download("https://drive.google.com/uc?id=1p-VIfl6ALTxdrbjo_OpDFxY6Pe__en0E", meme_data_path, quiet=False)

def save_image_locally(url, filename):
    """Save locally the image created from the data returned """
    # Save the meme locally
    r = requests.get(url, stream=True) #TODO: See if it's better to replace meme_data with meme_url
    if r.status_code == 200:
        with open(filename, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    return 

class MemeGenerator:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.api_root = "https://api.imgflip.com"
    def get_generatable_memes_info(self):
        with open("./meme_data.json") as f:
            res = json.load(f)
        return res
    def get_all_memes(self):
        """Get popular memes from the imgflip api"""
        data = requests.get(f"{self.api_root}/get_memes").json()
        return data['data']['memes']
    def create_meme(self, meme_id, text_list):
        """Create a custom meme with custom captions. Returns info from the api call."""
        # text_template = {f"text{i}": text for i, text in enumerate(text_list)}
        # boxes = [{"text": caption} for caption in text_list]
        boxes = {f"boxes[{i}][text]": text for i, text in enumerate(text_list)}
        data = {
            'template_id': meme_id, 
            'username': self.username,
            'password': self.password,
            **boxes
        }
        created_meme = requests.post(f"{self.api_root}/caption_image", data=data).json()
        return created_meme
    def save_local_meme(self, meme_data, filename):
        """Save locally the image created from the data returned """
        # Save the meme locally
        save_image_locally(meme_data['data']['url'], filename)
        return
    def get_initial_word_from_text(self, text, method):
        if method == "long_words":
            words = text.split(" ")
            important_words = [word for word in words if len(word) > 4]
            if len(important_words) > 0:
                return random.choice(important_words)
            else:    
                return ""
        elif method == "ntlk_verbs":
            # From https://stackoverflow.com/questions/5404243/extracting-english-verbs-from-a-given-text/5410074
            from nltk import word_tokenize, pos_tag, download
            # try:
                # nltk.data.find("tokenizers/punkt")
            # except LookupError:
            download("punkt", quiet=True)
            download("averaged_perceptron_tagger", quiet=True)
            tokens = word_tokenize(text)
            pos_tagged_tokens = pos_tag(tokens)
            verbs = [word[0] for word in pos_tagged_tokens if word[1].startswith('V')]
            if len(verbs) > 0:
                return random.choice(verbs)
            else:
                return "" 
        else:
            return ""
    def predict_meme_text(self, model_path, template_id, num_boxes, init_text = '', model_filename="model.h5", params_filename="params.json", beam_width=1, max_output_length=140):
        """
        Required: 
            - pretrained model
            - params.json: contains information like seq_length, mappings char_to_int. It should be automatically generated after running train.py
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
        from keras.models import load_model
        from keras.preprocessing.sequence import pad_sequences
        import json
        import util
        print("Generating text... This might take a minute.")
        model = load_model(os.path.join(model_path, model_filename))
        # model = transformers.TFAutoModelForSeq2SeqLM.from_pretrained("ralcanta/share-meme-generator", from_tf=True)
        params = json.load(open(os.path.join(model_path, params_filename)))
        SEQUENCE_LENGTH = params['sequence_length']
        char_to_int = params['char_to_int']
        labels = {v: k for k, v in params['labels_index'].items()}


        template_id = str(template_id).zfill(12)
        min_score = 0.1

        final_texts = [{'text': init_text, 'score': 1}]
        finished_texts = []
        for char_count in range(len(init_text), max_output_length):
            texts = []
            for i in range(0, len(final_texts)):
                box_index = str(final_texts[i]['text'].count('|'))
                texts.append(template_id + '  ' + box_index + '  ' + final_texts[i]['text'])
            sequences = util.texts_to_sequences(texts, char_to_int)
            data = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)
            predictions_list = model.predict(data)
            # print(predictions_list)
            sorted_predictions = []
            for i in range(0, len(predictions_list)):
                for j in range(0, len(predictions_list[i])):
                    sorted_predictions.append({
                        'text': final_texts[i]['text'],
                        'next_char': labels[j],
                        'score': predictions_list[i][j] * final_texts[i]['score']
                    })

            sorted_predictions = sorted(sorted_predictions, key=lambda p: p['score'], reverse=True)
            top_predictions = []
            top_score = sorted_predictions[0]['score']
            rand_int = random.randint(int(min_score * 1000), 1000)
            for prediction in sorted_predictions:
                # give each prediction a chance of being chosen corresponding to its score
                if prediction['score'] >= rand_int / 1000 * top_score:
                # or swap above line with this one to enable equal probabilities instead
                # if prediction['score'] >= top_score * min_score:
                    top_predictions.append(prediction)
            random.shuffle(top_predictions)
            final_texts = []
            for i in range(0, min(beam_width, len(top_predictions)) - len(finished_texts)):
                prediction = top_predictions[i]
                final_texts.append({
                    'text': prediction['text'] + prediction['next_char'],
                    # normalize all scores around top_score=1 so tiny floats don't disappear due to rounding
                    'score': prediction['score'] / top_score
                })
                if prediction['next_char'] == '|' and prediction['text'].count('|') == num_boxes - 1:
                    finished_texts.append(final_texts[len(final_texts) - 1])
                    final_texts.pop()

            if char_count >= max_output_length - 1 or len(final_texts) == 0:
                final_texts = final_texts + finished_texts
                final_texts = sorted(final_texts, key=lambda p: p['score'], reverse=True)
                return final_texts[0]['text']
    def generate_captions(self, meme_id, input_text, num_boxes):      
        """Uses MaCHinE LEarNiNG to create a caption *hopefully* related to it"""
        #STEP 1: Find an important word (for now lets just try a verb)
        initial_word = self.get_initial_word_from_text(input_text, method="ntlk_verbs") + " "
        download_model() #Download model if not downloaded already
        #STEP 2: Generate caption!
        generated_captions = self.predict_meme_text(
            model_path = ".", #should be downloaded in current directory
            template_id = meme_id, 
            num_boxes = num_boxes,  
            init_text = initial_word, 
            model_filename = "meme_generator_model.h5",
            params_filename = "params_generator_model.json",
            beam_width = 1, 
            max_output_length = 140
        )
        print(f"Generated captions: {generated_captions}. Now posting to imgflip.com...")
        captions = generated_captions.split("|")[:-1] #last one is empty
        meme_info = self.create_meme(meme_id, captions)
        return meme_info

    
import csv
import json 

memer = MemeGenerator("mit_meme_creator", "mit_meme_password")
popular = memer.get_all_memes()
# memer.generate_captions("93895088", "Hello I am working in this thing and I'm excited to grow.")
# memer.create_meme(93895088, ["Here I'd put real-people confessions", "If I only had one"])

with open("../ml-scripts/meme_text_gen_convnet/train_gena_24_4000.json") as f:
    data = json.load(f)

all_ids = set()

for _id, caption in data:
    all_ids.add(_id)

i = 0
final_info = {
    "info": "Meme info from the dataset https://www.kaggle.com/dylanwenzlau/imgflip-meme-text-samples-for-top-24-memes ",
    "memes": []
}
for meme in popular:
    _id = meme['id']
    if _id in all_ids:
        final_info['memes'].append({
            **meme, 
            'examples_url': f"https://imgflip.com/meme/{_id}"
        })
with open("meme_data.json", 'w') as f:
    json.dump(final_info, f, indent=4)

        