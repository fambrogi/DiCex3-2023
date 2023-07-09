# sudo docker container ls
# sudo docker stop 53eb099c0163
# sudo docker exec -t -i dic-assignment /bin/bash

sudo docker stop dic-assignment
sudo docker rm dic-assignment
sudo docker build -t dic-assignment .

docker run -v local_directory:/app -d -p 5000:5000 --network=host dic-assignment 
sudo docker exec -t -i dic-assignment /bin/bash

curl http://localhost:5000/api/detect -d "input=./images/in/"

sudo docker cp dic-assignment:/app/images/out/ ./images/
sudo docker cp dic-assignment:/app/inference_times.txt .
tail -1 inference_times.txt

# sudo curl http://localhost:5000/api/detect -d "input=./images/birds_test.jpg&output=1"
