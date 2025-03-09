# nav2py_template_Pas_CrowdNav

This branch provide the implementation of the Pas_CrowdNav algorithm in the Nav2py template.  
[Original Repository](https://github.com/yejimun/PaS_CrowdNav)

## 📌 Installation

### **1. Clone the Repository**
```bash
git clone git@github.com:zenghjian/PaS_CrowdNav.git /nav2py_pas_crowdnav_controller/nav2py_pas_crowdnav_controller
```

### **2. Install Dependencies**

Install the dependencies using the following command in the root directory of the planner repository:
```bash
pip install -e .
```

### **3. Model Path**

Due to conflict of ros2 environment, for simplicity, please put absolute path of the model in [here](https://github.com/zenghjian/Nav2_PaS_CrowdNav/blob/humble/nav2py_pas_crowdnav_controller/nav2py_pas_crowdnav_controller/__main__.py#L35).