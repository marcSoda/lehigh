FROM ubuntu
MAINTAINER Chase Mattingly (cmm323@lehigh.edu)

RUN apt-get update -y
RUN apt-get upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git man curl build-essential screen gdb libssl-dev psmisc python3

WORKDIR "/root"
