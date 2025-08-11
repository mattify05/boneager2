# :bone: Bone-Ager

**Bone-Ager** is an automatic paediatric bone assessment software designed to assist radiologists by providing efficient, accurate and consistent bone-age predictions from hand X-rays. It uses machine learning to detect and evaluate X-rays, helping to streamline clinical workflows and reduce diagnostic variability.

---

## :mag_right: Key Features 

- **Trained ML model:** High accuracy bone age prediction using the RSNA database
- **Fast results:** Inference time in under 5 seconds 
- **Local Execution:** Runs offline on desktop computers, without requiring internet connection 
- **User-friendly interface:** Simple design to enhance the workflow for clinicians 

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
### 1. Download the Bone-Ager model
- Download our pre-trained model from [this link](https://github.com/jjjaden-hash/DESN2000-BINF-M13B_GAMMA/blob/main/bone_age/best_bone_age_model.pth).
- Locate the file in your **Downloads** folder - it should be named _best_bone_age_model.pth_
- Drag this file to the cloned repository into the **bone_age** directory and replace the current model with the newly downloaded one.

(image1)

(image2)

(image3)
### 2. Runninng Bone-Ager
To get started with bone processing, type the following into your terminal:
```bash
cd bone_age/frontend
streamlit run main.py
```
- Note: Bone-ager uses the PyTorch library for machine learning. While this software is compatible with Windows, macOS and Linux, there
  will be slight variations in bone age predictions due to the different hardware and software system configurations for each user

---

## :bulb: Credits

Bone-Ager was developed by **Team M13B Gamma**, a student group studying Bioinformatics at the University of New South Wales. 
