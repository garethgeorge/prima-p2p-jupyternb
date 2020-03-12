docker build -t lastpenguin/jupyter-bionic-vision .
docker run -d --name jupyter --volume "$(pwd):/home/jovyan/work" -p 8888:8888 lastpenguin/jupyter-bionic-vision start-notebook.sh
