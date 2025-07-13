import roboflow

# Đăng nhập vào Roboflow
roboflow.login()  # Nhập API key khi được yêu cầu

# Tải dataset
rf = roboflow.Roboflow()
# Tải dataset 2: Original or Fake Shoes
project = rf.workspace("ggdgdsgdg").project("original-or-fake-shoes")
dataset = project.version(5).download(model_format="coco", location="original-or-fake-shoes")
print("Dataset 2 downloaded to:", dataset.location)

# Tải dataset 3: Shoe Authentication App
project = rf.workspace("authenticatorapp").project("shoe-authentication-app")
dataset = project.version(2).download(model_format="coco", location="shoe-authentication-app")
print("Dataset 3 downloaded to:", dataset.location)