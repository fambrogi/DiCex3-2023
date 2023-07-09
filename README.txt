#Build docker image from Dockerfile
docker build -t dic-assignment .

# Run docker container locally
docker run -v local_directory:/app -d -p 5000:5000 dic-assignment

#run inference on single image files from a path
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg"

curl http://localhost:5000/api/detect -d "input=./images/birds_test.jpg"


#to save annotated images, add a "flag" by adding any character to "output"
curl http://localhost:5000/api/detect -d "input=./images/filename.jpg&output=1"

curl http://localhost:5000/api/detect -d "input=./images/birds_test.jpg&output=1"




#####################################################################################
sudo docker build -t dic-assignment .

sudo docker run -v local_directory:/app -p 5000:5000 dic-assignment
sudo docker run -v local_directory:/app -d -p 5000:5000 dic-assignment
