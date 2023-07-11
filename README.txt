#Build docker image from Dockerfile
docker build .

#create docker image
docker build -t getting-started .

#run image:
docker run -d -p 8000:81 getting-started

#get docker status
docker ps

#stop docker image execution
getting-started 


#run inference on single image files from a path
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg"

curl http://localhost:5000/api/detect -d "input=./images/birds_test.jpg"


#to save annotated images, add a "flag" by adding any character to "output"
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg&output=1"

curl http://localhost:5000/api/detect -d "input=./images/birds_test.jpg&output=1"


