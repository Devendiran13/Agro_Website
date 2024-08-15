import cv2
import os
from werkzeug.utils import secure_filename
from flask import Flask,request,render_template,Markup
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
model = load_model('best_modelpest.h5')

pest_dic = {
    'Ants': """ <b>Pest</b>: Ants <br/>Biological Name:  Hymenoptera, family Formicidae <br/>
        <br/> Details about this Pest:

        <br/><br/>1.Ants are known for their highly organized and complex social structures. A colony typically consists of a queen, worker ants, and male ants. The queen is responsible for laying eggs, while worker ants perform various tasks such as foraging, caring for the young, and defending the nest.

        <br/> 2. Ants communicate primarily through pheromones, which are chemical signals that they release. These pheromones help in marking trails, signaling danger, and coordinating various activities within the colony.
        <br/><br/> Effects of Ants in Agriculture<br/>
        <br/>1. Soil Aeration: Ants burrow into the soil, creating channels that enhance aeration and water infiltration. This can improve overall soil structure and nutrient availability for plants.

        <br/>2.Seed Dispersal: Some ant species play a role in seed dispersal, helping with the distribution of seeds across a landscape, which can be beneficial for plant propagation.
        
        <br/>3. Pest Management: Ants can act as agricultural pests by farming and protecting honeydew-producing insects such as aphids. These insects feed on plant sap and excrete honeydew, which can attract ants. In return for protecting these pests from predators, ants feed on honeydew, potentially leading to increased pest populations and damage to crops.
        <br/>4. Seed Predation: Certain ant species may consume seeds or seedlings, impacting germination rates and the establishment of crops.""",

    'Bees': """ <b>Pest</b>: Bees  <br/>Biological Name:  Hymenoptera,  family Apidae<br/>
        <br/> Details about this Pest:

        <br/><br/>Bees are flying insects known for their crucial role in pollination and honey production. There are over 20,000 known species of bees, and they belong to the order Hymenoptera, which also includes ants and wasps. Bees play a vital role in ecosystems and agriculture, contributing to the pollination of flowering plants, which, in turn, facilitates the production of fruits, vegetables, and seeds.
        <br/><br/> Effects of Bees in Agriculture <br/>
        <br/>1. Bees contribute significantly to agriculture by enhancing the yield and quality of crops through pollination. Many fruits, vegetables, and nuts depend on bee pollination for successful reproduction.

        <br/>2. Despite their ecological and economic importance, bees face various threats, including habitat loss, pesticide exposure, diseases, and climate change. Conservation efforts are underway to protect and preserve bee populations for the benefit of ecosystems and agriculture.""",

    'Beetle': """ <b>Pest</b>: Beetle <br/>Biological Name:Leptinotarsa decemlineata <br/>
        <br/> Details about this Pest:

        <br/><br/>Beetles are a diverse group of insects, and while many species are harmless or even beneficial in ecosystems, some can have significant effects on agriculture, both positive and negative.

        <br/><br/> Effects of Beetle in Agriculture <br/>
        <br/>1. Crop Pests: Some beetles are agricultural pests that can damage crops by feeding on leaves, stems, roots, or fruits. Examples include the Colorado Potato Beetle, Mexican Bean Beetle, and various species of weevils.

        <br/>2. Pollination: Some beetles, including various species of flower beetles, can be important pollinators of flowering plants. While bees are the most well-known pollinators, beetles play a role in the pollination of certain crops and wildflowers.
        
        <br/>3. Decomposition: Some beetles are important in breaking down organic matter, contributing to nutrient cycling in ecosystems. """,



    'Caterpillar': """ <b>Pest</b>: Caterpillar <br/>Biological Name: Danaus plexippus<br/>
        <br/> Details about this Pest:

        <br/><br/>"Caterpillar" is a term used to refer to the larval stage of moths and butterflies. Caterpillars are characterized by their segmented bodies, multiple legs, and often, the presence of hairs or bristles. They are herbivores, feeding on leaves and other plant parts.

        <br/><br/> Effects of Caterpillar in Agriculture <br/>
        <br/>1. Damage to Crops: Caterpillars can be agricultural pests, causing damage to crops by feeding on leaves, stems, and fruits. Some caterpillar species are notorious for defoliating plants, reducing yields and affecting crop quality.

        <br/>2. Pest Management: Farmers may use various methods to control caterpillar populations, including the application of insecticides, introducing natural predators, and employing cultural practices like crop rotation.""",

    'Earthworms': """ <b>Pest</b>: Earthworms <br/>Biological Name: Lumbricus terrestris <br/>
        <br/> Details about this Pest:

        <br/><br/>Earthworms are segmented worms belonging to the class Oligochaeta. They play a crucial role in soil health and fertility.

        <br/><br/> Effects of Earthworms in Agriculture <br/>
        <br/>1. Soil Aeration: Earthworms create burrows in the soil, promoting aeration. This enhances the exchange of gases (such as oxygen and carbon dioxide) between the soil and the atmosphere, benefiting plant roots.

        <br/>2. Nutrient Cycling: Earthworms consume organic matter, breaking it down into nutrient-rich castings. These castings are a valuable source of nutrients for plants and contribute to soil fertility. 
        
        <br/>3. Farmers' Friend: Earthworms are considered beneficial to agriculture due to their positive impact on soil health. Practices that support earthworm populations, such as avoiding excessive use of chemical pesticides, can contribute to sustainable farming. """,

    'Earwig': """ <b>Pest</b>: Earwig <br/>Biological Name:Forficula auricularia <br/>
        <br/> Details about this Pest:

        <br/><br/>Earwigs belong to the order Dermaptera. They are characterized by elongated bodies, pincer-like appendages at the rear, and membranous wings. Earwigs are omnivores, feeding on both plant material and small insects.

        <br/><br/> Effects of Earwig in Agriculture <br/>
        <br/>1. Feeding on Plants: Earwigs can feed on plant materials, including flowers, fruits, and tender shoots. In some cases, they may damage crops by chewing on leaves or eating seedlings.

        <br/>2. Beneficial Predators: While earwigs can be pests in certain situations, they also feed on small insects and insect eggs, providing some level of natural pest control.""",
    

    'Grasshopper': """ <b>Pest</b>: Grasshopper <br/>Biological Name:Chorthippus brunneus  <br/>
        <br/> Details about this Pest:

        <br/><br/>Grasshoppers belong to the order Orthoptera. They are known for their strong hind legs adapted for jumping and their herbivorous feeding habits.

        <br/><br/> Effects of Grassshopper in Agriculture <br/>
        <br/>1. Feeding on Crops: Grasshoppers are voracious feeders and can cause significant damage to crops by consuming leaves, stems, and grains. High grasshopper populations can lead to crop loss and economic damage.

        <br/>2. Management: In agricultural settings, controlling grasshopper populations may involve the use of insecticides, habitat management, and biological control methods.""",
    'Moth': """ <b>Pest</b>: Moth <br/>Biological Name: Tineola bisselliella<br/>
        <br/> Details about this Pest:

        <br/><br/>Moths belong to the order Lepidoptera, and they are closely related to butterflies. Moths typically have antennae that are feathery or thread-like, and they often have a more robust body compared to butterflies.

        <br/><br/> Effects of Moth in Agriculture <br/>
        <br/>1. Crop Pest: While moths themselves are not the primary agricultural pests, their larvae (caterpillars) can be significant pests. Moth caterpillars may feed on crops, fruits, and leaves, causing damage similar to other caterpillar species.

        <br/>2. Integrated Pest Management: Moth pests are often managed through integrated pest management (IPM) practices, which may include biological control, cultural practices, and the judicious use of insecticides. """,

    'Slug': """ <b>Pest</b>: Slug <br/>Biological Name: Arion distinctus<br/>
        <br/> Details about this Pest:

        <br/><br/>Slugs are gastropod mollusks that lack a visible shell. They have a soft, slimy body and are commonly found in damp environments.

        <br/><br/> Effects of Slug in Agriculture <br/>
        <br/>1. Plant Feeding: Slugs are known for feeding on a variety of plant material, including leaves, stems, and seedlings. They can cause damage to crops, particularly in moist and humid conditions.
        
        <br/>3. Damage Control: Farmers may employ various methods to control slug populations, including the use of baits, traps, and cultural practices like removing hiding places.""",



    'Snail': """ <b>Pest</b>: Snail m<br/>Biological Name: Helix aspersa <br/>
        <br/> Details about this Pest:

        <br/><br/>Snails are also gastropod mollusks, but they have a spiral shell. Like slugs, they can be herbivorous or omnivorous.

        <br/><br/> Effects of Snail in Agriculture <br/>
        <br/>1. Plant Feeding: Snails can feed on a variety of plants, including crops, ornamental plants, and garden vegetables. Their feeding can lead to damage to leaves, fruits, and seedlings.

        <br/>2. Control Measures: Farmers and gardeners may use barriers, baits, and other control methods to manage snail populations and protect crops """,
    'Wasp': """ <b>Pest</b>: Wasp <br/>Biological Name: Vespula vulgaris <br/>
        <br/> Details about this Pest:

        <br/><br/>Wasps are flying insects belonging to the order Hymenoptera. They can be both predators and pollinators.

        <br/><br/> Effects of Wasp in Agriculture <br/>
        <br/>1. Predation: Some wasp species are beneficial predators that feed on other insects, including pest insects. They can contribute to natural pest control in agricultural ecosystems.

        <br/>3. Pollination: Certain wasp species play a role in pollination, although bees are generally more significant pollinators in agriculture.""",
   'Weevil': """ <b>Pest</b>: Weevil <br/>Biological Name: Sitophilus oryzae <br/>
        <br/> Details about this Pest:

        <br/><br/>Weevils are a type of beetle belonging to the superfamily Curculionoidea. They are characterized by their elongated snouts.

        <br/><br/> Effects of Weevil in Agriculture <br/>
        <br/>1. Stored Grain Pests: Weevils can be significant pests in stored grains, infesting and damaging stored cereals, rice, and other food products.

        <br/>2. Crop Damage: Some weevil species can also damage crops in the field by feeding on leaves, stems, and grains.
        
        <br/>3. Integrated Pest Management: Control measures for weevils include proper storage practices, sanitation, and the use of insecticides when necessary.""",
}


UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(path):
    ref = {0: 'Ants', 1: 'Bees', 2: 'Beetle', 3: 'Caterpillar', 4: 'Earthworms', 5: 'Earwig', 6: 'Grasshopper',
           7: 'Moth', 8: 'Slug', 9: 'Snail', 10: 'Wasp', 11: 'Weevil'}
    img = load_img(path,target_size=(224,224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)
    pred = np.argmax(model.predict(img))
    return ref[pred]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        pred = prediction(UPLOAD_FOLDER+'/'+filename)
        response = Markup(str(pest_dic[pred]))
        return render_template('home.html',org_img_name=filename,recommendation=response,prediction=pred)


if __name__ == '__main__':
    app.run(debug=True)