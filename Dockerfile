FROM pytorch/pytorch

# Install OpenFed
RUN git clone git@github.com:FederalLab/OpenFed.git /OpenFed
WORKDIR /OpenFed
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
