# :bone: Bone-Ager

**Bone-Ager** is an automatic paediatric bone assessment software designed to assist radiologists by providing efficient, accurate and consistent bone-age predictions from hand X-rays. It uses machine learning to detect and evaluate X-rays, helping to streamline clinical workflows and reduce diagnostic variability.

---

## :mag_right: Key Features 

- **Uses a trained model:** Highly accurate bone age prediction using the RSNA database
- **Fast inference time:** Returns results in under 5 seconds 
- **Local Execution:** Runs offline on desktop computers, without requiring internet connection 
- **User-friendly interface:** With a simple design to enhance the workflow for clinicians 

---

## :computer: Installation

### :file_folder: Requirements

- Python 3.11 
- pip
- See `requirements.txt` for the Python package dependencies 

### Setup

Clone the repository and install all dependencies:
```bash
git clone https://github.com/jjjaden-hash/DESN2000-BINF-M13B_GAMMA.git
cd bone_age
pip install -r requirements.txt
```

## Usage
After setting up the code on VSCode, head over to [**this link**](www.google.com) to download the BoneAger model.
Once it is installed, drag it from your Applications / Downloads folder into the
bone_age directory contained in your GitHub repo. Below are a few images that can
show you how to do this:

(image1)

(image2)

(image3)

To get started with bone processing, type the following into your terminal:
```bash
cd bone_age/frontend
streamlit run main.py
```

---

## :bulb: Credits

Bone-Ager was developed by **Team M13B Gamma**, a student group studying Bioinformatics at the University of New South Wales. 
