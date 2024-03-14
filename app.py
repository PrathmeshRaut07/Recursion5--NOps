import json
from urllib.request import urlopen
import folium
from folium.plugins import MarkerCluster
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import cv2
import tensorflow as tf
import os
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'Fight_Model.h5')
loaded_model_path = os.path.join(script_dir, 'fc_1.h5')
model = tf.keras.models.load_model(model_path)
loaded_model = tf.keras.models.load_model(loaded_model_path)
custom_data = [
    {
        "tag": "greeting",
        "patterns": [
            "Hi",
            "How are you",
            "Is anyone there?",
            "Hello",
            "Good day"
        ],
        "responses": [
            "Hello, thanks for visiting",
            "Good to see you again",
            "Hi there, how can I help?"
        ],
        "context_set": ""
    },
    {
        "tag": "goodbye",
        "patterns": [
            "Bye",
            "See you later",
            "Goodbye"
        ],
        "responses": [
            "See you later, thanks for visiting",
            "Have a nice day",
            "Bye! Come back again soon."
        ]
    },
    {
            "tag": "chatbot",
            "patterns": [
                "Who built this chatbot?",
                "Tell me about Chatbot",
                "What is this chatbot name?"
            ],
            "responses": [
                "Hi, I am Chatbot designed by Shobit.",
                "Thanks for asking. I am designed by Shobit.",
                "I am a chatbot."
            ]
        },
        {
            "tag": "about",
            "patterns": [
                "Who are you?",
                "Tell me about Yourself",
                "What is this?"
            ],
            "responses": [
                "Hi, I am User. Nice to meet you. I made this chatbot for fun and practice.",
                "Thanks for asking. I am Shobit Gupta, coder by profession but ML enthusiast by passion."
            ]
        },
        {
            "tag": "data_privacy",
            "patterns": [
                "How does AI ensure the privacy and protection of user data?",
                "What measures are in place to protect sensitive information?",
                "Is my data safe with AI systems?"
            ],
            "responses": [
                "AI systems employ encryption, anonymization techniques, and access controls to safeguard user data.",
                "Data privacy is a top priority for AI systems, with measures such as encryption and access controls implemented to protect sensitive information.",
                "Rest assured, AI systems are designed with robust security measures to ensure the privacy and protection of user data."
            ]
        },
        {
            "tag": "cybersecurity",
            "patterns": [
                "How does AI help in detecting and preventing cybersecurity threats?",
                "Can AI systems enhance our cybersecurity measures?",
                "What role does AI play in cybersecurity?"
            ],
            "responses": [
                "AI algorithms analyze vast amounts of data to detect anomalies and identify potential cybersecurity threats.",
                "AI enhances cybersecurity measures through proactive threat detection and response mechanisms.",
                "With AI's ability to analyze data in real-time, it significantly improves our cybersecurity posture by detecting and mitigating threats more effectively."
            ]
        },
        {
            "tag": "adversarial_attacks",
            "patterns": [
                "What are adversarial attacks?",
                "How does AI defend against adversarial attacks?",
                "Can AI systems be vulnerable to adversarial attacks?"
            ],
            "responses": [
                "Adversarial attacks are attempts to deceive or manipulate AI systems by exploiting vulnerabilities.",
                "AI systems defend against adversarial attacks through techniques like adversarial training and robust optimization.",
                "While AI systems can be vulnerable to adversarial attacks, ongoing research and advancements in security measures help mitigate these risks."
            ]
        },
        {
            "tag": "model_robustness",
            "patterns": [
                "How does AI ensure fairness and robustness in its models?",
                "What measures are taken to address biases in AI models?",
                "Can AI models be biased?"
            ],
            "responses": [
                "AI models undergo rigorous testing to detect biases and ensure fairness across different demographics.",
                "Techniques such as bias detection, fairness metrics, and diverse dataset curation are employed to mitigate biases and enhance model robustness.",
                "While AI models can be prone to biases, efforts are made to address these issues through fairness-aware algorithms and diverse dataset representation."
            ]
        },
        {
            "tag": "secure_development",
            "patterns": [
                "How is security integrated into the AI development lifecycle?",
                "What practices ensure secure AI development?",
                "Can AI systems be vulnerable to security breaches?"
            ],
            "responses": [
                "Security measures are incorporated into every stage of the AI development lifecycle, including requirements gathering, design, implementation, testing, and deployment.",
                "Practices such as secure coding standards, code reviews, and security testing help ensure the integrity and resilience of AI systems against potential security breaches.",
                "While AI systems can be vulnerable to security breaches, adherence to secure development practices mitigates these risks, making them more robust and secure."
            ]
        },
        {
            "tag": "ethics_and_regulations",
            "patterns": [
                "What ethical considerations are important in AI security?",
                "Are there regulations governing the use of AI in security?",
                "How does AI adhere to ethical guidelines in security applications?"
            ],
            "responses": [
                "Ethical considerations in AI security involve ensuring fairness, transparency, and accountability in AI systems.",
                "Regulations such as GDPR and CCPA govern the use of AI in security and mandate protection of user data and privacy.",
                "AI adheres to ethical guidelines in security applications by promoting transparency, fairness, and responsible use of technology to minimize risks and uphold user trust."
            ]
        },
        {
            "tag": "incident_response",
            "patterns": [
                "What is the process for incident response in AI security?",
                "How are security incidents handled in AI systems?",
                "Is there a protocol for responding to security breaches involving AI?"
            ],
            "responses": [
                "Incident response in AI security involves detecting, analyzing, and mitigating security incidents in AI systems.",
                "Security incidents in AI systems are handled through a structured response plan, involving identification of the breach, containment, eradication, recovery, and post-incident analysis.",
                "A protocol for responding to security breaches involving AI includes immediate action to contain the breach, investigation to determine the cause, remediation of vulnerabilities, and communication with stakeholders."
            ]
        }
    
]

def visualize(image, bounding_box, class_name='Fight', text_size=0.5, box_color=(255, 0, 0)):
    xmin, ymin, xmax, ymax = bounding_box
    font_size = 0.7
    thickness = 2
    color = box_color
    class_name_str = str(class_name)
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    # Ensure that the image has the correct data type
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # Ensure that the image is in RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(image_rgb, class_name_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness,
                cv2.LINE_AA)

    return image_rgb


def save_visualization(filename, bounding_box, class_name, save_path):
    image = cv2.imread(filename)
    img = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    result_image = visualize(img, bounding_box, class_name)
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, result_image_bgr)
    print(f"Result image saved to: {save_path}")


def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    image = img / 255
    img = np.expand_dims(image, axis=0)
    temp = loaded_model.predict(img)
    a = temp[0][0]
    if a < 0.5:
        annotations = [[0.5, 0.5, 0.5, 0.5]]
        return annotations, image
    else:
        annotations = model.predict(img)
        return annotations, image


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image-service', methods=['GET', 'POST'])
def image_service():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            prediction, image = predict_image(filename)
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')
            save_visualization(filename, prediction[0], 'Fight', result_path)

            return render_template('result.html', prediction=prediction[0], result_path='result_image.jpg')

        return 'Invalid file format. Allowed formats are png, jpg, jpeg, gif, mp4, mov, avi.'

    return render_template('index.html')

@app.route('/get_result_image')
def get_result_image():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')
    return send_file(result_path, mimetype='image/jpeg')


@app.route('/map')
def display_map():
    # Fetch location data
    url = 'http://ipinfo.io/json'
    response = urlopen(url)
    data = json.load(response)
    loc = data['loc']
    loc = loc.split(',')
    latitude = float(loc[0])
    longitude = float(loc[1])

    # Create map centered around user's location
    map_centered = folium.Map(location=[latitude, longitude], zoom_start=15)

    # Add marker cluster to the map
    marker_cluster = MarkerCluster().add_to(map_centered)

    # Add darker marker for the exact location to the marker cluster
    folium.Marker(
        location=[latitude, longitude],
        icon=folium.Icon(color='darkpurple', icon_color='white', icon='circle', prefix='fa')
    ).add_to(marker_cluster)

    # Save the map as HTML file
    map_centered.save('map.html')

    # Return the HTML file as response
    return send_file('map.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    # Encode the question using BART tokenizer
    inputs = tokenizer(question, return_tensors="pt")
    # Generate answer using the BART model
    generated_ids = model.generate(inputs.input_ids)
    # Decode the generated answer
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response_dict = {intent['tag']: intent['responses'] for intent in custom_data}
    # Check if the answer matches any known response in the custom data
    for intent in custom_data:
        for pattern in intent['patterns']:
            if pattern.lower() in question.lower():
                return jsonify({'answer': response_dict[intent['tag']][0]})  # Return the first response
                # You can also randomize the response if you have multiple responses

    return jsonify({'answer': answer})
'''
def display_map():
    # Fetch location data
    url = 'http://ipinfo.io/json'
    response = urlopen(url)
    data = json.load(response)
    loc = data['loc']
    loc = loc.split(',')
    latitude = float(loc[0])
    longitude = float(loc[1])
    map_centered = folium.Map(location=[latitude, longitude], zoom_start=10)
    map_centered.save('map.html')
    return send_file('map.html')
'''
if __name__ == '__main__':
    app.run(debug=True)

